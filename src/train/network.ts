import type { Tensor } from "../core/tensor.js";
import type { Kongyo2xModelJSON, ModelLayerJSON, ModelMeta } from "../model/types.js";
import { convNetworkJSON, IDENTITY_ALPHA, LEAKY_RELU_ALPHA } from "../brain/network.js";
import { convForward, convBackward, type ConvLayerParams } from "./conv.js";
import { deconvForward, deconvBackward, type DeconvLayerParams } from "./deconv.js";
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

/** The subset of a network the training loop drives, shared by every architecture. */
export interface TrainableNetwork {
  readonly offset: number;
  forward(input: Tensor): { output: Tensor };
  predict(input: Tensor): Tensor;
  backward(cache: { output: Tensor }, dOutput: Tensor): void;
  zeroGrad(): void;
  parameters(): Parameter[];
  snapshot(): Float32Array[];
  load(values: Float32Array[]): void;
  toModelJSON(meta: ModelMeta): Kongyo2xModelJSON;
}

function convParamsToJSON(layer: ConvLayerParams): ModelLayerJSON {
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

export class MlpconvNetwork implements TrainableNetwork {
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
    const layers: ModelLayerJSON[] = this.layers.map((layer) => convParamsToJSON(layer));
    return { type: "kongyo2x", version: 1, meta, layers };
  }
}

/** Conv-channel widths of the shipped `scale2.0x_model.json`; a sensible default upconv body. */
export const DEFAULT_UPCONV_CHANNELS: readonly number[] = [16, 32, 64, 128, 128, 256];

export interface UpconvCache {
  acts: Tensor[];
  pres: Tensor[];
  headInput: Tensor;
  output: Tensor;
}

/**
 * waifu2x `upconv_7`-style network: a stack of valid 3x3 convs with LeakyReLU(0.1),
 * followed by a 4x4 stride-2 transposed convolution that both refines and doubles the
 * resolution. Maps 1:1 onto the kongyo2x `upconv` model format (`resize: true`).
 */
export class UpconvNetwork implements TrainableNetwork {
  readonly convLayers: ConvLayerParams[];
  readonly head: DeconvLayerParams;
  readonly channels: number;
  readonly scale: number;

  private constructor(convLayers: ConvLayerParams[], head: DeconvLayerParams, channels: number, scale: number) {
    this.convLayers = convLayers;
    this.head = head;
    this.channels = channels;
    this.scale = scale;
  }

  static create(channels: number, convChannels: readonly number[], scale: number, rng: Rng): UpconvNetwork {
    if (scale !== 2) {
      throw new Error(`upconv architecture supports scale 2 only, received ${scale}`);
    }
    if (convChannels.length === 0) {
      throw new Error("upconv requires at least one conv layer");
    }
    const convLayers: ConvLayerParams[] = [];
    let cin = channels;
    for (const width of convChannels) {
      const kh = 3;
      const kw = 3;
      const fanIn = cin * kh * kw;
      const weights = new Float32Array(width * fanIn);
      heInit(weights, fanIn, LEAKY_RELU_ALPHA, rng);
      convLayers.push({
        cin,
        cout: width,
        kh,
        kw,
        alpha: LEAKY_RELU_ALPHA,
        weights,
        bias: new Float32Array(width),
        gradWeights: new Float32Array(width * fanIn),
        gradBias: new Float32Array(width),
      });
      cin = width;
    }
    const kh = 4;
    const kw = 4;
    const headWeights = new Float32Array(cin * channels * kh * kw);
    linearInit(headWeights, channels * kh * kw, rng);
    const head: DeconvLayerParams = {
      cin,
      cout: channels,
      kh,
      kw,
      strideX: scale,
      strideY: scale,
      padX: scale + 1,
      padY: scale + 1,
      adjX: 0,
      adjY: 0,
      weights: headWeights,
      bias: new Float32Array(channels),
      gradWeights: new Float32Array(cin * channels * kh * kw),
      gradBias: new Float32Array(channels),
    };
    return new UpconvNetwork(convLayers, head, channels, scale);
  }

  get offset(): number {
    return 2 * this.convLayers.length + 2;
  }

  forward(input: Tensor): UpconvCache {
    const acts: Tensor[] = [input];
    const pres: Tensor[] = [];
    let current = input;
    for (const layer of this.convLayers) {
      const pre = convForward(current, layer);
      pres.push(pre);
      current = leakyRelu(pre, layer.alpha);
      acts.push(current);
    }
    const output = deconvForward(current, this.head);
    return { acts, pres, headInput: current, output };
  }

  predict(input: Tensor): Tensor {
    return this.forward(input).output;
  }

  backward(cache: UpconvCache, dOutput: Tensor): void {
    let d = deconvBackward(cache.headInput, this.head, dOutput, true) as Tensor;
    for (let l = this.convLayers.length - 1; l >= 0; l--) {
      const layer = this.convLayers[l] as ConvLayerParams;
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
    for (const layer of this.convLayers) {
      layer.gradWeights.fill(0);
      layer.gradBias.fill(0);
    }
    this.head.gradWeights.fill(0);
    this.head.gradBias.fill(0);
  }

  parameters(): Parameter[] {
    const params: Parameter[] = [];
    for (const layer of this.convLayers) {
      params.push({ value: layer.weights, grad: layer.gradWeights });
      params.push({ value: layer.bias, grad: layer.gradBias });
    }
    params.push({ value: this.head.weights, grad: this.head.gradWeights });
    params.push({ value: this.head.bias, grad: this.head.gradBias });
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
    const layers: ModelLayerJSON[] = this.convLayers.map((layer) => convParamsToJSON(layer));
    const head = this.head;
    layers.push({
      kind: "deconv",
      inputPlanes: head.cin,
      outputPlanes: head.cout,
      kernelWidth: head.kw,
      kernelHeight: head.kh,
      strideX: head.strideX,
      strideY: head.strideY,
      padX: head.padX,
      padY: head.padY,
      adjX: head.adjX,
      adjY: head.adjY,
      weights: Array.from(head.weights),
      bias: Array.from(head.bias),
    });
    return { type: "kongyo2x", version: 1, meta, layers };
  }
}
