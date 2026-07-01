import { buildNetwork } from "../brain/network.js";
import type { BrainNetworkJSON, BrainNeuralNetwork } from "../brain/network.js";
import type {
  ConvLayer,
  ConvLayerJSON,
  DeconvLayer,
  DeconvLayerJSON,
  Kongyo2xModelJSON,
  ModelLayer,
  ModelLayerJSON,
  ModelMeta,
} from "./types.js";

function tryBuildNetwork(json: BrainNetworkJSON): BrainNeuralNetwork | undefined {
  try {
    return buildNetwork(json);
  } catch {
    return undefined;
  }
}

function convFromJSON(layer: ConvLayerJSON): ConvLayer {
  const sizes = layer.network.sizes;
  const inputSize = sizes[0] ?? 0;
  const outputPlanes = sizes[1] ?? 0;
  const kernelArea = layer.kernelWidth * layer.kernelHeight;
  const inputPlanes = kernelArea > 0 ? inputSize / kernelArea : 0;
  if (!Number.isInteger(inputPlanes) || inputPlanes <= 0 || !Number.isInteger(outputPlanes) || outputPlanes <= 0) {
    throw new Error(
      `conv layer network sizes [${sizes.join(", ")}] do not fit a ${layer.kernelWidth}x${layer.kernelHeight} kernel`,
    );
  }
  return {
    kind: "conv",
    inputPlanes,
    outputPlanes,
    kernelWidth: layer.kernelWidth,
    kernelHeight: layer.kernelHeight,
    strideX: layer.strideX,
    strideY: layer.strideY,
    padX: layer.padX,
    padY: layer.padY,
  };
}

function deconvFromJSON(layer: DeconvLayerJSON): DeconvLayer {
  const expectedWeights = layer.inputPlanes * layer.outputPlanes * layer.kernelWidth * layer.kernelHeight;
  if (layer.weights.length !== expectedWeights || layer.bias.length !== layer.outputPlanes) {
    throw new Error(
      `deconv layer expects ${expectedWeights} weights and ${layer.outputPlanes} biases, ` +
        `received ${layer.weights.length} and ${layer.bias.length}`,
    );
  }
  return {
    kind: "deconv",
    inputPlanes: layer.inputPlanes,
    outputPlanes: layer.outputPlanes,
    kernelWidth: layer.kernelWidth,
    kernelHeight: layer.kernelHeight,
    strideX: layer.strideX,
    strideY: layer.strideY,
    padX: layer.padX,
    padY: layer.padY,
    adjX: layer.adjX,
    adjY: layer.adjY,
    weights: Float32Array.from(layer.weights),
    bias: Float32Array.from(layer.bias),
  };
}

function deconvToJSON(layer: DeconvLayer): DeconvLayerJSON {
  return {
    kind: "deconv",
    inputPlanes: layer.inputPlanes,
    outputPlanes: layer.outputPlanes,
    kernelWidth: layer.kernelWidth,
    kernelHeight: layer.kernelHeight,
    strideX: layer.strideX,
    strideY: layer.strideY,
    padX: layer.padX,
    padY: layer.padY,
    adjX: layer.adjX,
    adjY: layer.adjY,
    weights: Array.from(layer.weights),
    bias: Array.from(layer.bias),
  };
}

export interface ConvParams {
  weights: Float32Array;
  bias: Float32Array;
  alpha: number;
}

/**
 * Extract raw dense-layer parameters from a serialized brain.js network, when
 * the network has the exact shape the fast engines understand: a single
 * fully-connected layer with a leaky-relu activation.
 */
function convParamsFromJSON(stored: ConvLayerJSON): ConvParams | undefined {
  const network = stored.network;
  if (network.options.activation !== "leaky-relu" || network.trainOpts.activation !== "leaky-relu") {
    return undefined;
  }
  if (network.layers.length !== 2) {
    return undefined;
  }
  const dense = network.layers[1];
  if (!dense) {
    return undefined;
  }
  const rows = dense.weights;
  const outputPlanes = rows.length;
  const inputSize = outputPlanes > 0 ? (rows[0]?.length ?? 0) : 0;
  if (outputPlanes !== (network.sizes[1] ?? 0) || inputSize !== (network.sizes[0] ?? 0)) {
    return undefined;
  }
  const weights = new Float32Array(outputPlanes * inputSize);
  for (let o = 0; o < outputPlanes; o++) {
    const row = rows[o] ?? [];
    for (let k = 0; k < inputSize; k++) {
      weights[o * inputSize + k] = row[k] ?? 0;
    }
  }
  return {
    weights,
    bias: Float32Array.from(dense.biases),
    alpha: network.trainOpts.leakyReluAlpha,
  };
}

export class Kongyo2xModel {
  readonly meta: ModelMeta;
  readonly layers: ModelLayer[];
  private readonly convJSON: Array<ConvLayerJSON | undefined>;
  /** Lazily built brain.js networks; `null` marks an attempted build that failed. */
  private readonly convNets: Array<BrainNeuralNetwork | null | undefined>;
  /** Lazily extracted raw parameters; `null` marks a layer without raw parameters. */
  private readonly params: Array<ConvParams | null | undefined>;

  private constructor(meta: ModelMeta, layers: ModelLayer[], convJSON: Array<ConvLayerJSON | undefined>) {
    this.meta = meta;
    this.layers = layers;
    this.convJSON = convJSON;
    this.convNets = new Array<BrainNeuralNetwork | null | undefined>(layers.length);
    this.params = new Array<ConvParams | null | undefined>(layers.length);
  }

  static fromJSON(json: Kongyo2xModelJSON): Kongyo2xModel {
    if (json.type !== "kongyo2x") {
      throw new Error(`unrecognized model type ${String(json.type)} (expected "kongyo2x")`);
    }
    if (!Array.isArray(json.layers) || json.layers.length === 0) {
      throw new Error("kongyo2x model must contain at least one layer");
    }
    const layers: ModelLayer[] = [];
    const convJSON: Array<ConvLayerJSON | undefined> = [];
    for (const layer of json.layers) {
      if (layer.kind === "deconv") {
        layers.push(deconvFromJSON(layer));
        convJSON.push(undefined);
      } else {
        layers.push(convFromJSON(layer));
        convJSON.push(layer);
      }
    }
    return new Kongyo2xModel(json.meta, layers, convJSON);
  }

  toJSON(): Kongyo2xModelJSON {
    const layers: ModelLayerJSON[] = this.layers.map((layer, index) => {
      if (layer.kind === "deconv") {
        return deconvToJSON(layer);
      }
      const stored = this.convJSON[index] as ConvLayerJSON;
      return {
        kind: "conv",
        kernelWidth: layer.kernelWidth,
        kernelHeight: layer.kernelHeight,
        strideX: layer.strideX,
        strideY: layer.strideY,
        padX: layer.padX,
        padY: layer.padY,
        network: stored.network,
      };
    });
    return { type: "kongyo2x", version: 1, meta: this.meta, layers };
  }

  /**
   * brain.js network for a conv layer, built on first use. Returns undefined
   * for deconv layers and when brain.js is not installed.
   */
  convNetwork(index: number): BrainNeuralNetwork | undefined {
    const stored = this.convJSON[index];
    if (!stored) {
      return undefined;
    }
    let net = this.convNets[index];
    if (net === undefined) {
      net = tryBuildNetwork(stored.network) ?? null;
      this.convNets[index] = net;
    }
    return net ?? undefined;
  }

  /**
   * Raw conv parameters for the fast engines, extracted once per layer and
   * cached. The returned arrays are shared across calls — treat them as
   * immutable.
   */
  convParams(index: number): ConvParams | undefined {
    let cached = this.params[index];
    if (cached === undefined) {
      const stored = this.convJSON[index];
      cached = stored ? (convParamsFromJSON(stored) ?? null) : null;
      this.params[index] = cached;
    }
    return cached ?? undefined;
  }

  get isRgb(): boolean {
    return this.meta.channels === 3;
  }
}
