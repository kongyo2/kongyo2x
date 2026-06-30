import type { Tensor } from "../core/tensor.js";
import type { ModelMeta } from "../model/types.js";
import { crop } from "../image/pad.js";
import { createRng, type Rng } from "./rng.js";
import { MlpconvNetwork, mlpconvSpecs } from "./network.js";
import { Adam, Ema, clipGradNorm, cosineLr } from "./optimizer.js";
import { computeLoss, meanSquaredError, psnr, type LossKind } from "./loss.js";
import { degrade, type Dataset, type Degradation } from "./data.js";

export interface TrainConfig {
  channels: number;
  width: number;
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
  loss: LossKind;
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
  network: MlpconvNetwork;
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
  inputCrop: Tensor;
}

export function defaultTrainConfig(overrides: Partial<TrainConfig> = {}): TrainConfig {
  return {
    channels: 1,
    width: 20,
    degradation: { scale: 2 },
    iterations: 600,
    batchSize: 8,
    patchSize: 40,
    baseLr: 1e-3,
    minLr: 1e-5,
    warmup: 60,
    weightDecay: 1e-5,
    gradClip: 1.0,
    emaDecay: 0.999,
    loss: "charbonnier",
    seed: 1234,
    logEvery: 100,
    valBatch: 16,
    ...overrides,
  };
}

function centerTarget(clean: Tensor, offset: number): Tensor {
  return crop(clean, offset, offset, clean.width - offset, clean.height - offset);
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
  const rng: Rng = createRng(config.seed);
  const network = MlpconvNetwork.create(config.channels, mlpconvSpecs(config.channels, config.width), rng);
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

  const valRng: Rng = createRng(config.seed + 7919);
  const valItems: ValItem[] = [];
  for (let i = 0; i < config.valBatch; i++) {
    const clean = dataset.sample(config.patchSize, valRng);
    const input = degrade(clean, config.degradation);
    valItems.push({
      input,
      target: centerTarget(clean, offset),
      inputCrop: centerTarget(input, offset),
    });
  }

  const baselinePsnr = psnr(average(valItems.map((item) => meanSquaredError(item.inputCrop, item.target))));
  const evalMse = (): number =>
    average(valItems.map((item) => meanSquaredError(network.predict(item.input), item.target)));

  const initialValPsnr = psnr(evalMse());
  const history: TrainProgress[] = [];

  for (let step = 0; step < config.iterations; step++) {
    const lr = cosineLr(step, schedule);
    network.zeroGrad();
    let lossSum = 0;
    for (let b = 0; b < config.batchSize; b++) {
      const clean = dataset.sample(config.patchSize, rng);
      const input = degrade(clean, config.degradation);
      const cache = network.forward(input);
      const target = centerTarget(clean, offset);
      const { loss, grad } = computeLoss(cache.output, target, config.loss);
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
    archName: "mlpconv",
    channels: config.channels,
    offset,
    scaleFactor: config.degradation.scale,
    resize: false,
  };

  return { network, meta, history, finalValPsnr: psnr(finalMse), initialValPsnr, baselinePsnr, usedEma };
}
