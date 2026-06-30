import { loadBrain } from "../brain/loader.js";
import type { BrainNeuralNetwork } from "../brain/loader.js";
import type { ConvLayer, DeconvLayer, ModelLayer, ModelMeta, RawLayerJSON, RawModelJSON } from "./types.js";

const LEAKY_RELU_ALPHA = 0.1;
const IDENTITY_ALPHA = 1.0;

function flattenConvWeights(layer: RawLayerJSON): Float32Array {
  const { nOutputPlane, nInputPlane, kH, kW } = layer;
  const out = new Float32Array(nOutputPlane * nInputPlane * kH * kW);
  let idx = 0;
  for (let o = 0; o < nOutputPlane; o++) {
    const wo = layer.weight[o] as number[][][];
    for (let i = 0; i < nInputPlane; i++) {
      const wi = wo[i] as number[][];
      for (let ky = 0; ky < kH; ky++) {
        const row = wi[ky] as number[];
        for (let kx = 0; kx < kW; kx++) {
          out[idx++] = row[kx] as number;
        }
      }
    }
  }
  return out;
}

function flattenDeconvWeights(layer: RawLayerJSON): Float32Array {
  const { nInputPlane, nOutputPlane, kH, kW } = layer;
  const out = new Float32Array(nInputPlane * nOutputPlane * kH * kW);
  let idx = 0;
  for (let i = 0; i < nInputPlane; i++) {
    const wi = layer.weight[i] as number[][][];
    for (let o = 0; o < nOutputPlane; o++) {
      const wo = wi[o] as number[][];
      for (let ky = 0; ky < kH; ky++) {
        const row = wo[ky] as number[];
        for (let kx = 0; kx < kW; kx++) {
          out[idx++] = row[kx] as number;
        }
      }
    }
  }
  return out;
}

function makeBias(layer: RawLayerJSON): Float32Array {
  const bias = new Float32Array(layer.nOutputPlane);
  if (layer.bias) {
    for (let o = 0; o < layer.nOutputPlane; o++) {
      bias[o] = layer.bias[o] ?? 0;
    }
  }
  return bias;
}

function toModelLayer(layer: RawLayerJSON): ModelLayer {
  const strideX = layer.dW ?? 1;
  const strideY = layer.dH ?? 1;
  const padX = layer.padW ?? 0;
  const padY = layer.padH ?? 0;
  if (layer.class_name === "nn.SpatialFullConvolution") {
    const deconv: DeconvLayer = {
      kind: "deconv",
      inputPlanes: layer.nInputPlane,
      outputPlanes: layer.nOutputPlane,
      kernelWidth: layer.kW,
      kernelHeight: layer.kH,
      strideX,
      strideY,
      padX,
      padY,
      adjX: layer.adjW ?? 0,
      adjY: layer.adjH ?? 0,
      weights: flattenDeconvWeights(layer),
      bias: makeBias(layer),
    };
    return deconv;
  }
  const conv: ConvLayer = {
    kind: "conv",
    inputPlanes: layer.nInputPlane,
    outputPlanes: layer.nOutputPlane,
    kernelWidth: layer.kW,
    kernelHeight: layer.kH,
    strideX,
    strideY,
    padX,
    padY,
    weights: flattenConvWeights(layer),
    bias: makeBias(layer),
  };
  return conv;
}

function buildBrainConv(layer: ConvLayer, alpha: number): BrainNeuralNetwork {
  const { NeuralNetwork } = loadBrain();
  const inSize = layer.inputPlanes * layer.kernelHeight * layer.kernelWidth;
  const outSize = layer.outputPlanes;
  const weights: number[][] = [];
  for (let o = 0; o < outSize; o++) {
    const row = new Array<number>(inSize);
    const base = o * inSize;
    for (let k = 0; k < inSize; k++) {
      row[k] = layer.weights[base + k] as number;
    }
    weights.push(row);
  }
  const biases = Array.from(layer.bias);
  const net = new NeuralNetwork();
  net.fromJSON({
    type: "NeuralNetwork",
    sizes: [inSize, outSize],
    layers: [
      { weights: [], biases: [] },
      { weights, biases },
    ],
    inputLookup: null,
    inputLookupLength: 0,
    outputLookup: null,
    outputLookupLength: 0,
    options: {
      inputSize: inSize,
      outputSize: outSize,
      binaryThresh: 0.5,
      activation: "leaky-relu",
      leakyReluAlpha: LEAKY_RELU_ALPHA,
      hiddenLayers: [],
    },
    trainOpts: { activation: "leaky-relu", leakyReluAlpha: LEAKY_RELU_ALPHA, timeout: "Infinity" },
  });
  net.trainOpts.leakyReluAlpha = alpha;
  return net;
}

function deriveMeta(raw: RawModelJSON, layers: ModelLayer[]): ModelMeta {
  const config = raw[0]?.model_config;
  const lastLayer = layers[layers.length - 1];
  const hasDeconv = layers.some((l) => l.kind === "deconv");
  const fallbackChannels = lastLayer ? lastLayer.outputPlanes : 1;
  const fallbackOffset = layers
    .filter((l): l is Extract<ModelLayer, { kind: "conv" }> => l.kind === "conv")
    .reduce((acc, l) => acc + Math.floor((l.kernelWidth - 1) / 2), 0);
  const scaleFactor = config?.scale_factor ?? (hasDeconv ? 2 : 1);
  return {
    archName: config?.arch_name ?? (hasDeconv ? "upconv_7" : "vgg_7"),
    channels: config?.channels ?? fallbackChannels,
    offset: config?.offset ?? fallbackOffset,
    scaleFactor,
    resize: config?.resize ?? scaleFactor > 1,
  };
}

export class Waifu2xModel {
  readonly meta: ModelMeta;
  readonly layers: ModelLayer[];
  private readonly brainLayers: Array<BrainNeuralNetwork | undefined>;

  private constructor(meta: ModelMeta, layers: ModelLayer[], brainLayers: Array<BrainNeuralNetwork | undefined>) {
    this.meta = meta;
    this.layers = layers;
    this.brainLayers = brainLayers;
  }

  static fromJSON(raw: RawModelJSON): Waifu2xModel {
    if (!Array.isArray(raw) || raw.length === 0) {
      throw new Error("waifu2x model JSON must be a non-empty array of layers");
    }
    const layers = raw.map(toModelLayer);
    const meta = deriveMeta(raw, layers);
    const brainLayers = layers.map((layer, index) => {
      if (layer.kind !== "conv") {
        return undefined;
      }
      const isLast = index === layers.length - 1;
      return buildBrainConv(layer, isLast ? IDENTITY_ALPHA : LEAKY_RELU_ALPHA);
    });
    return new Waifu2xModel(meta, layers, brainLayers);
  }

  isLastLayer(index: number): boolean {
    return index === this.layers.length - 1;
  }

  brainLayer(index: number): BrainNeuralNetwork | undefined {
    return this.brainLayers[index];
  }

  get isRgb(): boolean {
    return this.meta.channels === 3;
  }
}
