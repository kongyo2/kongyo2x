export type LayerClass = "nn.SpatialConvolutionMM" | "nn.SpatialFullConvolution";

export interface ModelConfig {
  arch_name?: string;
  scale_factor?: number;
  channels?: number;
  offset?: number;
  resize?: boolean;
}

export interface RawLayerJSON {
  class_name?: string;
  nInputPlane: number;
  nOutputPlane: number;
  kW: number;
  kH: number;
  dW?: number;
  dH?: number;
  padW?: number;
  padH?: number;
  adjW?: number;
  adjH?: number;
  weight: number[][][][];
  bias?: number[];
  model_config?: ModelConfig;
}

export type RawModelJSON = RawLayerJSON[];

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
  /** Flattened weights in [outPlane][inPlane * kH * kW] row-major order. */
  weights: Float32Array;
  bias: Float32Array;
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

export interface ModelMeta {
  archName: string;
  channels: number;
  offset: number;
  scaleFactor: number;
  resize: boolean;
}
