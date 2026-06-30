import type { BrainNetworkJSON } from "../brain/network.js";

export interface ModelMeta {
  archName: string;
  channels: number;
  offset: number;
  scaleFactor: number;
  resize: boolean;
}

export interface ConvLayer {
  kind: "conv";
  inputPlanes: number;
  outputPlanes: number;
  kernelWidth: number;
  kernelHeight: number;
  strideX: number;
  strideY: number;
  padX: number;
  padY: number;
}

export interface DeconvLayer {
  kind: "deconv";
  inputPlanes: number;
  outputPlanes: number;
  kernelWidth: number;
  kernelHeight: number;
  strideX: number;
  strideY: number;
  padX: number;
  padY: number;
  adjX: number;
  adjY: number;
  /** Raw weights indexed as weight[inPlane][outPlane][ky][kx]. */
  weights: Float32Array;
  bias: Float32Array;
}

export type ModelLayer = ConvLayer | DeconvLayer;

export interface ConvLayerJSON {
  kind: "conv";
  kernelWidth: number;
  kernelHeight: number;
  strideX: number;
  strideY: number;
  padX: number;
  padY: number;
  /** A brain.js NeuralNetwork serialization: one fully-connected layer over the im2col patch. */
  network: BrainNetworkJSON;
}

export interface DeconvLayerJSON {
  kind: "deconv";
  inputPlanes: number;
  outputPlanes: number;
  kernelWidth: number;
  kernelHeight: number;
  strideX: number;
  strideY: number;
  padX: number;
  padY: number;
  adjX: number;
  adjY: number;
  /** Flattened weights indexed as weight[inPlane][outPlane][ky][kx]. */
  weights: number[];
  bias: number[];
}

export type ModelLayerJSON = ConvLayerJSON | DeconvLayerJSON;

export interface Kongyo2xModelJSON {
  type: "kongyo2x";
  version: 1;
  meta: ModelMeta;
  layers: ModelLayerJSON[];
}
