import { describe, expect, it } from "vitest";
import { createTensor } from "../src/core/tensor.js";
import type { Tensor } from "../src/core/tensor.js";
import { convForward, type ConvLayerParams } from "../src/train/conv.js";
import { leakyRelu } from "../src/train/activation.js";
import { MlpconvNetwork, type LayerSpec, type Parameter } from "../src/train/network.js";
import { computeLoss } from "../src/train/loss.js";
import { Adam, Ema, clipGradNorm, cosineLr } from "../src/train/optimizer.js";
import { createRng } from "../src/train/rng.js";
import { trainModel, defaultTrainConfig } from "../src/train/trainer.js";
import { SyntheticDataset, degrade } from "../src/train/data.js";
import { brainConvForward } from "../src/engine/brainConv.js";
import { buildNetwork, convNetworkJSON } from "../src/brain/network.js";
import { runModel } from "../src/engine/runModel.js";
import { Kongyo2xModel } from "../src/model/model.js";
import type { ConvLayer } from "../src/model/types.js";
import { maxAbsDiff } from "./helpers.js";

function randomConv(cin: number, cout: number, kh: number, kw: number, alpha: number, seed: number): ConvLayerParams {
  const rng = createRng(seed);
  const k = cin * kh * kw;
  const weights = new Float32Array(cout * k);
  for (let i = 0; i < weights.length; i++) {
    weights[i] = (rng.next() - 0.5) * 0.8;
  }
  const bias = new Float32Array(cout);
  for (let o = 0; o < cout; o++) {
    bias[o] = (rng.next() - 0.5) * 0.2;
  }
  return {
    cin,
    cout,
    kh,
    kw,
    alpha,
    weights,
    bias,
    gradWeights: new Float32Array(cout * k),
    gradBias: new Float32Array(cout),
  };
}

function randomTensor(channels: number, height: number, width: number, seed: number): Tensor {
  const rng = createRng(seed);
  const t = createTensor(channels, height, width);
  for (let i = 0; i < t.data.length; i++) {
    t.data[i] = rng.next();
  }
  return t;
}

describe("conv forward matches brain.js", () => {
  for (const alpha of [0.1, 1.0]) {
    it(`reproduces brain.js output for alpha=${alpha}`, () => {
      const cin = 3;
      const cout = 4;
      const layer = randomConv(cin, cout, 3, 3, alpha, 42);
      const input = randomTensor(cin, 8, 9, 7);
      const mine = leakyRelu(convForward(input, layer), alpha);

      const k = cin * 3 * 3;
      const weights2D: number[][] = [];
      for (let o = 0; o < cout; o++) {
        weights2D.push(Array.from(layer.weights.subarray(o * k, (o + 1) * k)));
      }
      const net = buildNetwork(convNetworkJSON(weights2D, Array.from(layer.bias), alpha));
      const convLayer: ConvLayer = {
        kind: "conv",
        inputPlanes: cin,
        outputPlanes: cout,
        kernelWidth: 3,
        kernelHeight: 3,
        strideX: 1,
        strideY: 1,
        padX: 0,
        padY: 0,
      };
      const reference = brainConvForward(input, convLayer, net);
      expect(maxAbsDiff(mine, reference)).toBeLessThan(1e-4);
    });
  }
});

describe("analytical gradients match finite differences", () => {
  it("passes a numerical gradient check", () => {
    const specs: LayerSpec[] = [
      { kh: 3, kw: 3, cout: 4 },
      { kh: 1, kw: 1, cout: 3 },
      { kh: 1, kw: 1, cout: 1 },
    ];
    const net = MlpconvNetwork.create(1, specs, createRng(11));
    const input = randomTensor(1, 7, 7, 3);
    const target = randomTensor(1, 5, 5, 99);

    const evalLoss = (): number => computeLoss(net.predict(input), target, "mse").loss;

    net.zeroGrad();
    const cache = net.forward(input);
    const { grad } = computeLoss(cache.output, target, "mse");
    net.backward(cache, grad);

    const params = net.parameters();
    const probes: Array<[number, number]> = [
      [0, 0],
      [0, 7],
      [1, 0],
      [2, 1],
      [3, 0],
      [4, 2],
      [5, 0],
    ];
    const eps = 1e-3;
    for (const [pj, idx] of probes) {
      const param = params[pj] as Parameter;
      const original = param.value[idx] as number;
      param.value[idx] = original + eps;
      const lossPlus = evalLoss();
      param.value[idx] = original - eps;
      const lossMinus = evalLoss();
      param.value[idx] = original;
      const numeric = (lossPlus - lossMinus) / (2 * eps);
      const analytic = param.grad[idx] as number;
      expect(Math.abs(numeric - analytic)).toBeLessThan(1e-3 + 2e-2 * Math.abs(analytic));
    }
  });
});

describe("adam", () => {
  it("minimizes a quadratic toward the target", () => {
    const target = Float32Array.from([0.3, -0.7, 1.2, 0.0]);
    const value = new Float32Array(target.length);
    const grad = new Float32Array(target.length);
    const param: Parameter = { value, grad };
    const adam = new Adam([param]);
    for (let step = 0; step < 600; step++) {
      for (let i = 0; i < value.length; i++) {
        grad[i] = 2 * ((value[i] as number) - (target[i] as number));
      }
      adam.update(0.05);
    }
    expect(
      maxAbsDiff({ channels: 1, height: 1, width: 4, data: value }, { channels: 1, height: 1, width: 4, data: target }),
    ).toBeLessThan(1e-2);
  });
});

describe("gradient clipping and schedule", () => {
  it("clips by global norm", () => {
    const grad = Float32Array.from([3, 4]);
    const params: Parameter[] = [{ value: new Float32Array(2), grad }];
    const norm = clipGradNorm(params, 1);
    expect(norm).toBeCloseTo(5, 5);
    expect(Math.hypot(grad[0] as number, grad[1] as number)).toBeCloseTo(1, 5);
  });

  it("warms up then decays the learning rate", () => {
    const schedule = { baseLr: 1e-3, minLr: 1e-5, warmup: 10, totalSteps: 100 };
    expect(cosineLr(0, schedule)).toBeLessThan(cosineLr(9, schedule));
    expect(cosineLr(9, schedule)).toBeCloseTo(1e-3, 6);
    expect(cosineLr(99, schedule)).toBeLessThan(cosineLr(50, schedule));
  });

  it("smooths weights with a warmup-corrected EMA", () => {
    const value = Float32Array.from([1, 1]);
    const params: Parameter[] = [{ value, grad: new Float32Array(2) }];
    const ema = new Ema(params, 0.5);
    value.set([3, 3]);
    ema.update(params);
    value.set([0, 0]);
    ema.writeInto(params);
    const warmupDecay = 2 / 11;
    expect(value[0]).toBeCloseTo(warmupDecay * 1 + (1 - warmupDecay) * 3, 4);
  });
});

describe("exported model runs identically through the inference engine", () => {
  it("matches MlpconvNetwork.predict after serialization", () => {
    const specs: LayerSpec[] = [
      { kh: 3, kw: 3, cout: 4 },
      { kh: 1, kw: 1, cout: 4 },
      { kh: 3, kw: 3, cout: 1 },
    ];
    const net = MlpconvNetwork.create(1, specs, createRng(5));
    const input = randomTensor(1, 12, 12, 8);
    const direct = net.predict(input);

    const meta = { archName: "mlpconv", channels: 1, offset: net.offset, scaleFactor: 1, resize: false };
    const model = Kongyo2xModel.fromJSON(net.toModelJSON(meta));
    const viaEngine = runModel(model, input);
    expect(maxAbsDiff(direct, viaEngine)).toBeLessThan(1e-4);
  });
});

describe("degrade", () => {
  it("preserves channel count and dimensions for multi-channel input", () => {
    const clean = randomTensor(3, 6, 8, 4);
    const low = degrade(clean, { scale: 2 });
    expect(low.channels).toBe(3);
    expect(low.height).toBe(6);
    expect(low.width).toBe(8);
    expect(maxAbsDiff(low, clean)).toBeGreaterThan(0);
  });
});

describe("end-to-end training", () => {
  it("improves over the untrained model and reduces the loss", () => {
    const config = defaultTrainConfig({
      degradation: { scale: 2 },
      iterations: 120,
      batchSize: 4,
      patchSize: 32,
      width: 12,
      warmup: 20,
      logEvery: 40,
      valBatch: 6,
      seed: 7,
    });
    const result = trainModel(config, new SyntheticDataset());
    expect(result.history.length).toBeGreaterThan(0);
    const first = result.history[0]?.loss as number;
    const last = result.history[result.history.length - 1]?.loss as number;
    expect(last).toBeLessThan(first);
    expect(result.finalValPsnr).toBeGreaterThan(result.initialValPsnr + 3);
  });
});
