import type { Tensor } from "../core/tensor.js";
import { createTensor } from "../core/tensor.js";
import type { ModelMeta } from "../model/types.js";
import { crop } from "../image/pad.js";
import { rgb2y } from "../image/color.js";
import { resizeLanczos, resizeNearest } from "../image/resize.js";
import { createRng, type Rng } from "./rng.js";
import {
  MlpconvNetwork,
  UpconvNetwork,
  mlpconvSpecs,
  DEFAULT_UPCONV_CHANNELS,
  type TrainableNetwork,
} from "./network.js";
import { Adam, Ema, clipGradNorm, cosineLr } from "./optimizer.js";
import { computeLoss, meanSquaredError, psnr, type LossKind } from "./loss.js";
import { type Dataset, type Degradation } from "./data.js";

export type TrainArch = "upconv" | "mlpconv";
export type TrainLoss = LossKind | "mix";

export interface TrainConfig {
  arch: TrainArch;
  channels: number;
  width: number;
  convChannels: readonly number[];
  degradation: Degradation;
  iterations: number;
  batchSize: number;
  patchSize: number;
  baseLr: number;
  minLr: number;
  warmup: number;
  weightDecay: number;
  gradClip: number;
  emaDecay: number;
  loss: TrainLoss;
  mseFinetuneFrac: number;
  charbonnierEps: number;
  seed: number;
  logEvery: number;
  valBatch: number;
}

export interface TrainProgress {
  step: number;
  loss: number;
  lr: number;
  gradNorm: number;
  valPsnr: number;
  baselinePsnr: number;
}

export interface TrainResult {
  network: TrainableNetwork;
  meta: ModelMeta;
  history: TrainProgress[];
  finalValPsnr: number;
  initialValPsnr: number;
  baselinePsnr: number;
  usedEma: boolean;
}

interface ValItem {
  input: Tensor;
  target: Tensor;
  baseline: Tensor;
}

export function defaultTrainConfig(overrides: Partial<TrainConfig> = {}): TrainConfig {
  const arch: TrainArch = overrides.arch ?? "upconv";
  const upconv = arch === "upconv";
  return {
    arch,
    channels: upconv ? 3 : 1,
    width: 20,
    convChannels: DEFAULT_UPCONV_CHANNELS,
    degradation: { scale: 2 },
    iterations: 600,
    batchSize: 8,
    patchSize: upconv ? 64 : 40,
    baseLr: upconv ? 2e-4 : 1e-3,
    minLr: 1e-6,
    warmup: 60,
    weightDecay: 1e-5,
    gradClip: 1.0,
    emaDecay: 0.999,
    loss: upconv ? "mix" : "charbonnier",
    mseFinetuneFrac: 0.4,
    charbonnierEps: 1e-3,
    seed: 1234,
    logEvery: 100,
    valBatch: 16,
    ...overrides,
  };
}

function buildNetwork(config: TrainConfig, rng: Rng): TrainableNetwork {
  if (config.arch === "upconv") {
    return UpconvNetwork.create(config.channels, config.convChannels, config.degradation.scale, rng);
  }
  return MlpconvNetwork.create(config.channels, mlpconvSpecs(config.channels, config.width), rng);
}

function centerTarget(clean: Tensor, offset: number): Tensor {
  return crop(clean, offset, offset, clean.width - offset, clean.height - offset);
}

function downscale(clean: Tensor, scale: number): Tensor {
  return resizeLanczos(
    clean,
    Math.max(1, Math.round(clean.width / scale)),
    Math.max(1, Math.round(clean.height / scale)),
  );
}

/** The tensor fed to the network: the low-res patch itself for upconv, or a nearest-neighbour pre-upscale for mlpconv. */
function networkInput(lr: Tensor, arch: TrainArch, targetW: number, targetH: number): Tensor {
  return arch === "upconv" ? lr : resizeNearest(lr, targetW, targetH);
}

function toLuma(t: Tensor): Tensor {
  return t.channels === 3 ? rgb2y(t) : t;
}

function lumaMse(a: Tensor, b: Tensor): number {
  return meanSquaredError(toLuma(a), toLuma(b));
}

function flipLeftRight(t: Tensor): Tensor {
  const out = createTensor(t.channels, t.height, t.width);
  const w = t.width;
  for (let c = 0; c < t.channels; c++) {
    for (let y = 0; y < t.height; y++) {
      const row = (c * t.height + y) * w;
      for (let x = 0; x < w; x++) {
        out.data[row + x] = t.data[row + (w - 1 - x)] as number;
      }
    }
  }
  return out;
}

function flipTopBottom(t: Tensor): Tensor {
  const out = createTensor(t.channels, t.height, t.width);
  const { width: w, height: h } = t;
  for (let c = 0; c < t.channels; c++) {
    for (let y = 0; y < h; y++) {
      const src = (c * h + (h - 1 - y)) * w;
      const dst = (c * h + y) * w;
      for (let x = 0; x < w; x++) {
        out.data[dst + x] = t.data[src + x] as number;
      }
    }
  }
  return out;
}

function augment(clean: Tensor, rng: Rng): Tensor {
  let out = clean;
  if (rng.next() < 0.5) {
    out = flipLeftRight(out);
  }
  if (rng.next() < 0.5) {
    out = flipTopBottom(out);
  }
  return out;
}

function scaleInPlace(data: Float32Array, factor: number): void {
  for (let i = 0; i < data.length; i++) {
    data[i] = (data[i] as number) * factor;
  }
}

function average(values: number[]): number {
  let sum = 0;
  for (const value of values) {
    sum += value;
  }
  return values.length > 0 ? sum / values.length : 0;
}

export function trainModel(
  config: TrainConfig,
  dataset: Dataset,
  onProgress?: (progress: TrainProgress) => void,
): TrainResult {
  if (config.channels !== 1 && config.channels !== 3) {
    throw new Error(`unsupported channels ${config.channels} (expected 1 for luma or 3 for rgb)`);
  }
  const scale = config.degradation.scale;
  if (config.arch === "upconv" && config.patchSize % scale !== 0) {
    throw new Error(`patch size ${config.patchSize} must be divisible by scale ${scale} for upconv`);
  }
  const rng: Rng = createRng(config.seed);
  const network = buildNetwork(config, rng);
  const offset = network.offset;
  if (config.patchSize <= 2 * offset) {
    throw new Error(`patch size ${config.patchSize} too small for offset ${offset}`);
  }

  const params = network.parameters();
  const adam = new Adam(params, { weightDecay: config.weightDecay });
  const ema = new Ema(params, config.emaDecay);
  const schedule = {
    baseLr: config.baseLr,
    minLr: config.minLr,
    warmup: config.warmup,
    totalSteps: config.iterations,
  };
  const mseStart = Math.floor(config.iterations * (1 - config.mseFinetuneFrac));
  const lossAt = (step: number): LossKind => {
    if (config.loss === "mix") {
      return step >= mseStart ? "mse" : "charbonnier";
    }
    return config.loss;
  };

  const valRng: Rng = createRng(config.seed + 7919);
  const valItems: ValItem[] = [];
  for (let i = 0; i < config.valBatch; i++) {
    const clean = dataset.sample(config.patchSize, valRng);
    if (i === 0 && clean.channels !== config.channels) {
      throw new Error(
        `dataset produced ${clean.channels}-channel samples but the model expects ${config.channels}; ` +
          `construct the dataset with ${config.channels} channels`,
      );
    }
    const lr = downscale(clean, scale);
    valItems.push({
      input: networkInput(lr, config.arch, clean.width, clean.height),
      target: centerTarget(clean, offset),
      baseline: centerTarget(resizeLanczos(lr, clean.width, clean.height), offset),
    });
  }

  const baselinePsnr = psnr(average(valItems.map((item) => lumaMse(item.baseline, item.target))));
  const evalMse = (): number => average(valItems.map((item) => lumaMse(network.predict(item.input), item.target)));

  const initialValPsnr = psnr(evalMse());
  const history: TrainProgress[] = [];

  for (let step = 0; step < config.iterations; step++) {
    const lr = cosineLr(step, schedule);
    const kind = lossAt(step);
    network.zeroGrad();
    let lossSum = 0;
    for (let b = 0; b < config.batchSize; b++) {
      const clean = augment(dataset.sample(config.patchSize, rng), rng);
      const low = downscale(clean, scale);
      const input = networkInput(low, config.arch, clean.width, clean.height);
      const cache = network.forward(input);
      const target = centerTarget(clean, offset);
      const { loss, grad } = computeLoss(cache.output, target, kind, config.charbonnierEps);
      lossSum += loss;
      scaleInPlace(grad.data, 1 / config.batchSize);
      network.backward(cache, grad);
    }
    const gradNorm = clipGradNorm(params, config.gradClip);
    adam.update(lr);
    ema.update(params);

    if (config.logEvery > 0 && ((step + 1) % config.logEvery === 0 || step === config.iterations - 1)) {
      const progress: TrainProgress = {
        step: step + 1,
        loss: lossSum / config.batchSize,
        lr,
        gradNorm,
        valPsnr: psnr(evalMse()),
        baselinePsnr,
      };
      history.push(progress);
      onProgress?.(progress);
    }
  }

  const currentSnapshot = network.snapshot();
  const currentMse = evalMse();
  let usedEma = false;
  let finalMse = currentMse;
  if (ema.enabled) {
    ema.writeInto(params);
    const emaMse = evalMse();
    if (emaMse < currentMse) {
      usedEma = true;
      finalMse = emaMse;
    } else {
      network.load(currentSnapshot);
    }
  }

  const meta: ModelMeta = {
    archName: config.arch === "upconv" ? "kongyo2x_upconv" : "mlpconv",
    channels: config.channels,
    offset,
    scaleFactor: scale,
    resize: config.arch === "upconv",
  };

  return { network, meta, history, finalValPsnr: psnr(finalMse), initialValPsnr, baselinePsnr, usedEma };
}
