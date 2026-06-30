import { buildNetwork } from "../brain/network.js";
import type { BrainNeuralNetwork } from "../brain/network.js";
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

function convFromJSON(layer: ConvLayerJSON): ConvLayer {
  const sizes = layer.network.sizes;
  const inputSize = sizes[0] ?? 0;
  const outputPlanes = sizes[1] ?? 0;
  const inputPlanes =
    layer.kernelWidth * layer.kernelHeight > 0 ? inputSize / (layer.kernelWidth * layer.kernelHeight) : 0;
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

export class Kongyo2xModel {
  readonly meta: ModelMeta;
  readonly layers: ModelLayer[];
  private readonly convNets: Array<BrainNeuralNetwork | undefined>;
  private readonly convJSON: Array<ConvLayerJSON | undefined>;

  private constructor(
    meta: ModelMeta,
    layers: ModelLayer[],
    convNets: Array<BrainNeuralNetwork | undefined>,
    convJSON: Array<ConvLayerJSON | undefined>,
  ) {
    this.meta = meta;
    this.layers = layers;
    this.convNets = convNets;
    this.convJSON = convJSON;
  }

  static fromJSON(json: Kongyo2xModelJSON): Kongyo2xModel {
    if (json.type !== "kongyo2x") {
      throw new Error(`unrecognized model type ${String(json.type)} (expected "kongyo2x")`);
    }
    if (!Array.isArray(json.layers) || json.layers.length === 0) {
      throw new Error("kongyo2x model must contain at least one layer");
    }
    const layers: ModelLayer[] = [];
    const convNets: Array<BrainNeuralNetwork | undefined> = [];
    const convJSON: Array<ConvLayerJSON | undefined> = [];
    for (const layer of json.layers) {
      if (layer.kind === "deconv") {
        layers.push(deconvFromJSON(layer));
        convNets.push(undefined);
        convJSON.push(undefined);
      } else {
        layers.push(convFromJSON(layer));
        convNets.push(buildNetwork(layer.network));
        convJSON.push(layer);
      }
    }
    return new Kongyo2xModel(json.meta, layers, convNets, convJSON);
  }

  toJSON(): Kongyo2xModelJSON {
    const layers: ModelLayerJSON[] = this.layers.map((layer, index) => {
      if (layer.kind === "deconv") {
        return deconvToJSON(layer);
      }
      const stored = this.convJSON[index];
      const net = this.convNets[index];
      const network = net ? net.toJSON() : (stored as ConvLayerJSON).network;
      return {
        kind: "conv",
        kernelWidth: layer.kernelWidth,
        kernelHeight: layer.kernelHeight,
        strideX: layer.strideX,
        strideY: layer.strideY,
        padX: layer.padX,
        padY: layer.padY,
        network,
      };
    });
    return { type: "kongyo2x", version: 1, meta: this.meta, layers };
  }

  isLastLayer(index: number): boolean {
    return index === this.layers.length - 1;
  }

  convNetwork(index: number): BrainNeuralNetwork | undefined {
    return this.convNets[index];
  }

  get isRgb(): boolean {
    return this.meta.channels === 3;
  }
}
