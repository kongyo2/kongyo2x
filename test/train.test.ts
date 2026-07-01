import { describe, expect, it } from "vitest";
import { createTensor } from "../src/core/tensor.js";
import type { Tensor } from "../src/core/tensor.js";
import { convForward, type ConvLayerParams } from "../src/train/conv.js";
import { deconvForward, deconvBackward, type DeconvLayerParams } from "../src/train/deconv.js";
import { leakyRelu } from "../src/train/activation.js";
import { MlpconvNetwork, UpconvNetwork, type LayerSpec, type Parameter } from "../src/train/network.js";
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
import { brainAvailable, maxAbsDiff } from "./helpers.js";

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

describe.skipIf(!brainAvailable())("conv forward matches brain.js", () => {
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

  it("matches UpconvNetwork.predict after serialization", () => {
    const net = UpconvNetwork.create(3, [4, 6], 2, createRng(5));
    const input = randomTensor(3, 12, 12, 8);
    const direct = net.predict(input);

    const meta = { archName: "kongyo2x_upconv", channels: 3, offset: net.offset, scaleFactor: 2, resize: true };
    const model = Kongyo2xModel.fromJSON(net.toModelJSON(meta));
    expect(model.meta.resize).toBe(true);
    expect(model.layers.map((l) => l.kind)).toEqual(["conv", "conv", "deconv"]);
    const viaEngine = runModel(model, input);
    expect(direct.channels).toBe(3);
    expect([direct.height, direct.width]).toEqual([12, 12]);
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

describe("deconv analytical gradients match finite differences", () => {
  it("passes a numerical gradient check for a 4x4 stride-2 transposed conv", () => {
    const cin = 2;
    const cout = 3;
    const kh = 4;
    const kw = 4;
    const rng = createRng(21);
    const size = cin * cout * kh * kw;
    const weights = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      weights[i] = (rng.next() - 0.5) * 0.6;
    }
    const bias = Float32Array.from([0.1, -0.2, 0.05]);
    const layer: DeconvLayerParams = {
      cin,
      cout,
      kh,
      kw,
      strideX: 2,
      strideY: 2,
      padX: 3,
      padY: 3,
      adjX: 0,
      adjY: 0,
      weights,
      bias,
      gradWeights: new Float32Array(size),
      gradBias: new Float32Array(cout),
    };
    const input = randomTensor(cin, 5, 5, 3);
    const probe = deconvForward(input, layer);
    const target = randomTensor(cout, probe.height, probe.width, 99);

    const lossOf = (): number => {
      const out = deconvForward(input, layer);
      let sum = 0;
      for (let i = 0; i < out.data.length; i++) {
        const d = (out.data[i] as number) - (target.data[i] as number);
        sum += d * d;
      }
      return sum / out.data.length;
    };

    const out = deconvForward(input, layer);
    const n = out.data.length;
    const dOut = createTensor(cout, out.height, out.width);
    for (let i = 0; i < n; i++) {
      dOut.data[i] = (2 * ((out.data[i] as number) - (target.data[i] as number))) / n;
    }
    layer.gradWeights.fill(0);
    layer.gradBias.fill(0);
    const dInput = deconvBackward(input, layer, dOut, true) as Tensor;

    const eps = 1e-3;
    const check = (read: () => number, write: (v: number) => void, analytic: number): void => {
      const original = read();
      write(original + eps);
      const plus = lossOf();
      write(original - eps);
      const minus = lossOf();
      write(original);
      const numeric = (plus - minus) / (2 * eps);
      expect(Math.abs(numeric - analytic)).toBeLessThan(1e-3 + 2e-2 * Math.abs(analytic));
    };

    for (const idx of [0, 5, 17, 40, size - 1]) {
      check(
        () => weights[idx] as number,
        (v) => (weights[idx] = v),
        layer.gradWeights[idx] as number,
      );
    }
    for (let o = 0; o < cout; o++) {
      check(
        () => bias[o] as number,
        (v) => (bias[o] = v),
        layer.gradBias[o] as number,
      );
    }
    for (const idx of [0, 7, 24]) {
      check(
        () => input.data[idx] as number,
        (v) => (input.data[idx] = v),
        dInput.data[idx] as number,
      );
    }
  });
});

describe("upconv network gradients match finite differences", () => {
  it("passes a numerical gradient check through the deconv head and conv stack", () => {
    const net = UpconvNetwork.create(3, [4, 5], 2, createRng(11));
    const input = randomTensor(3, 10, 10, 3);
    const probe = net.predict(input);
    const target = randomTensor(3, probe.height, probe.width, 42);

    const evalLoss = (): number => computeLoss(net.predict(input), target, "mse").loss;

    net.zeroGrad();
    const cache = net.forward(input);
    const { grad } = computeLoss(cache.output, target, "mse");
    net.backward(cache, grad);

    const params = net.parameters();
    const probes: Array<[number, number]> = [
      [0, 0],
      [0, 5],
      [1, 1],
      [2, 3],
      [3, 0],
      [params.length - 2, 0],
      [params.length - 1, 1],
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
      expect(Math.abs(numeric - analytic)).toBeLessThan(1e-3 + 3e-2 * Math.abs(analytic));
    }
  });
});

describe("end-to-end training", () => {
  it("improves an mlpconv model over the untrained baseline and reduces the loss", () => {
    const config = defaultTrainConfig({
      arch: "mlpconv",
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
    const result = trainModel(config, new SyntheticDataset(1));
    expect(result.meta.resize).toBe(false);
    expect(result.history.length).toBeGreaterThan(0);
    const first = result.history[0]?.loss as number;
    const last = result.history[result.history.length - 1]?.loss as number;
    expect(last).toBeLessThan(first);
    expect(result.finalValPsnr).toBeGreaterThan(result.initialValPsnr + 3);
  });

  it("trains an rgb upconv model and reduces the loss", () => {
    const config = defaultTrainConfig({
      arch: "upconv",
      channels: 3,
      convChannels: [8, 16],
      degradation: { scale: 2 },
      iterations: 150,
      batchSize: 4,
      patchSize: 24,
      warmup: 20,
      logEvery: 50,
      valBatch: 6,
      seed: 7,
    });
    const result = trainModel(config, new SyntheticDataset(3));
    expect(result.meta.archName).toBe("kongyo2x_upconv");
    expect(result.meta.resize).toBe(true);
    expect(result.meta.channels).toBe(3);
    expect(result.meta.offset).toBe(6);
    const first = result.history[0]?.loss as number;
    const last = result.history[result.history.length - 1]?.loss as number;
    expect(last).toBeLessThan(first);
    expect(result.finalValPsnr).toBeGreaterThan(result.initialValPsnr + 1);

    const model = Kongyo2xModel.fromJSON(result.network.toModelJSON(result.meta));
    expect(model.layers.map((l) => l.kind)).toEqual(["conv", "conv", "deconv"]);
  });
});

describe("trainer input validation", () => {
  const tinyUpconv = (channels: number) =>
    defaultTrainConfig({
      arch: "upconv",
      channels,
      convChannels: [8],
      patchSize: 24,
      iterations: 1,
      valBatch: 1,
      logEvery: 0,
    });

  it("rejects unsupported channel counts", () => {
    expect(() => trainModel(tinyUpconv(2), new SyntheticDataset(2))).toThrow(/unsupported channels/);
  });

  it("rejects a dataset whose channel count does not match the model", () => {
    expect(() => trainModel(tinyUpconv(3), new SyntheticDataset(1))).toThrow(/expects 3/);
  });
});
