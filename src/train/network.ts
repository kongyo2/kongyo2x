import type { Tensor } from "../core/tensor.js";
import type { Kongyo2xModelJSON, ModelLayerJSON, ModelMeta } from "../model/types.js";
import { convNetworkJSON, IDENTITY_ALPHA, LEAKY_RELU_ALPHA } from "../brain/network.js";
import { convForward, convBackward, type ConvLayerParams } from "./conv.js";
import { leakyRelu, leakyReluGradInto } from "./activation.js";
import { heInit, linearInit } from "./init.js";
import { createTensor } from "../core/tensor.js";
import type { Rng } from "./rng.js";

export interface LayerSpec {
  kh: number;
  kw: number;
  cout: number;
}

export interface Parameter {
  readonly value: Float32Array;
  readonly grad: Float32Array;
}

export interface ForwardCache {
  acts: Tensor[];
  pres: Tensor[];
  output: Tensor;
}

export function mlpconvSpecs(channels: number, width = 32): LayerSpec[] {
  const wide = width;
  const narrow = Math.max(channels, Math.round(width / 2));
  return [
    { kh: 3, kw: 3, cout: narrow },
    { kh: 1, kw: 1, cout: narrow },
    { kh: 3, kw: 3, cout: wide },
    { kh: 1, kw: 1, cout: wide },
    { kh: 3, kw: 3, cout: wide },
    { kh: 1, kw: 1, cout: narrow },
    { kh: 1, kw: 1, cout: channels },
  ];
}

export class MlpconvNetwork {
  readonly layers: ConvLayerParams[];
  readonly channels: number;

  private constructor(layers: ConvLayerParams[], channels: number) {
    this.layers = layers;
    this.channels = channels;
  }

  static create(channels: number, specs: LayerSpec[], rng: Rng): MlpconvNetwork {
    const layers: ConvLayerParams[] = [];
    let cin = channels;
    for (let index = 0; index < specs.length; index++) {
      const spec = specs[index] as LayerSpec;
      const isLast = index === specs.length - 1;
      const alpha = isLast ? IDENTITY_ALPHA : LEAKY_RELU_ALPHA;
      const fanIn = cin * spec.kh * spec.kw;
      const weights = new Float32Array(spec.cout * fanIn);
      if (isLast) {
        linearInit(weights, fanIn, rng);
      } else {
        heInit(weights, fanIn, alpha, rng);
      }
      layers.push({
        cin,
        cout: spec.cout,
        kh: spec.kh,
        kw: spec.kw,
        alpha,
        weights,
        bias: new Float32Array(spec.cout),
        gradWeights: new Float32Array(spec.cout * fanIn),
        gradBias: new Float32Array(spec.cout),
      });
      cin = spec.cout;
    }
    return new MlpconvNetwork(layers, channels);
  }

  get offset(): number {
    let total = 0;
    for (const layer of this.layers) {
      total += (layer.kh - 1) / 2;
    }
    return total;
  }

  forward(input: Tensor): ForwardCache {
    const acts: Tensor[] = [input];
    const pres: Tensor[] = [];
    let current = input;
    for (const layer of this.layers) {
      const pre = convForward(current, layer);
      pres.push(pre);
      current = leakyRelu(pre, layer.alpha);
      acts.push(current);
    }
    return { acts, pres, output: current };
  }

  predict(input: Tensor): Tensor {
    return this.forward(input).output;
  }

  backward(cache: ForwardCache, dOutput: Tensor): void {
    let d: Tensor = dOutput;
    for (let l = this.layers.length - 1; l >= 0; l--) {
      const layer = this.layers[l] as ConvLayerParams;
      const pre = cache.pres[l] as Tensor;
      const input = cache.acts[l] as Tensor;
      const dPre = createTensor(pre.channels, pre.height, pre.width);
      leakyReluGradInto(dPre, pre, d, layer.alpha);
      const dInput = convBackward(input, layer, dPre, l > 0);
      if (dInput) {
        d = dInput;
      }
    }
  }

  zeroGrad(): void {
    for (const layer of this.layers) {
      layer.gradWeights.fill(0);
      layer.gradBias.fill(0);
    }
  }

  parameters(): Parameter[] {
    const params: Parameter[] = [];
    for (const layer of this.layers) {
      params.push({ value: layer.weights, grad: layer.gradWeights });
      params.push({ value: layer.bias, grad: layer.gradBias });
    }
    return params;
  }

  snapshot(): Float32Array[] {
    return this.parameters().map((p) => Float32Array.from(p.value));
  }

  load(values: Float32Array[]): void {
    const params = this.parameters();
    for (let i = 0; i < params.length; i++) {
      (params[i] as Parameter).value.set(values[i] as Float32Array);
    }
  }

  toModelJSON(meta: ModelMeta): Kongyo2xModelJSON {
    const layers: ModelLayerJSON[] = this.layers.map((layer) => {
      const rowLen = layer.cin * layer.kh * layer.kw;
      const weights: number[][] = [];
      for (let o = 0; o < layer.cout; o++) {
        weights.push(Array.from(layer.weights.subarray(o * rowLen, (o + 1) * rowLen)));
      }
      return {
        kind: "conv",
        kernelWidth: layer.kw,
        kernelHeight: layer.kh,
        strideX: 1,
        strideY: 1,
        padX: 0,
        padY: 0,
        network: convNetworkJSON(weights, Array.from(layer.bias), layer.alpha),
      };
    });
    return { type: "kongyo2x", version: 1, meta, layers };
  }
}
