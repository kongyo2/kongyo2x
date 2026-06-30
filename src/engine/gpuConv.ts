import { createRequire } from "node:module";
import { createTensor } from "../core/tensor.js";
import type { Tensor } from "../core/tensor.js";
import type { ConvLayer } from "../model/types.js";
import { convOutputSize } from "./conv.js";

interface KernelThis {
  thread: { x: number };
  constants: {
    inputPlanes: number;
    inH: number;
    inW: number;
    kH: number;
    kW: number;
    strideX: number;
    strideY: number;
    padX: number;
    padY: number;
    outW: number;
    planeSize: number;
    weightStride: number;
    alpha: number;
  };
}

type ConvKernel = (input: Float32Array, weights: Float32Array, bias: Float32Array) => ArrayLike<number>;

interface KernelSettings {
  output: [number];
  constants: KernelThis["constants"];
  loopMaxIterations: number;
  precision: "single";
}

interface GpuInstance {
  createKernel(
    fn: (this: KernelThis, input: Float32Array, weights: Float32Array, bias: Float32Array) => number,
    settings: KernelSettings,
  ): ConvKernel;
}

interface GpuModule {
  GPU: new (options?: { mode?: string }) => GpuInstance;
}

function convKernel(this: KernelThis, input: Float32Array, weights: Float32Array, bias: Float32Array): number {
  const t = this.thread.x;
  const o = Math.floor(t / this.constants.planeSize);
  const rem = t - o * this.constants.planeSize;
  const oy = Math.floor(rem / this.constants.outW);
  const ox = rem - oy * this.constants.outW;

  let sum = bias[o] as number;
  let w = o * this.constants.weightStride;
  for (let i = 0; i < this.constants.inputPlanes; i++) {
    const planeBase = i * this.constants.inH * this.constants.inW;
    for (let ky = 0; ky < this.constants.kH; ky++) {
      const iy = oy * this.constants.strideY - this.constants.padY + ky;
      for (let kx = 0; kx < this.constants.kW; kx++) {
        const ix = ox * this.constants.strideX - this.constants.padX + kx;
        let v = 0;
        if (iy >= 0 && iy < this.constants.inH && ix >= 0 && ix < this.constants.inW) {
          v = input[planeBase + iy * this.constants.inW + ix] as number;
        }
        sum = sum + v * (weights[w] as number);
        w = w + 1;
      }
    }
  }
  return Math.max(sum, this.constants.alpha * sum);
}

let gpu: GpuInstance | undefined;

function getGpu(): GpuInstance {
  if (!gpu) {
    const require = createRequire(import.meta.url);
    const { GPU } = require("gpu.js") as GpuModule;
    gpu = new GPU();
  }
  return gpu;
}

const kernels = new Map<string, ConvKernel>();

function kernelFor(layer: ConvLayer, inH: number, inW: number, outH: number, outW: number, alpha: number): ConvKernel {
  const weightStride = layer.inputPlanes * layer.kernelHeight * layer.kernelWidth;
  const key = [
    layer.inputPlanes,
    layer.outputPlanes,
    inH,
    inW,
    layer.kernelHeight,
    layer.kernelWidth,
    layer.strideX,
    layer.strideY,
    layer.padX,
    layer.padY,
    alpha,
  ].join(":");
  const existing = kernels.get(key);
  if (existing) {
    return existing;
  }
  const kernel = getGpu().createKernel(convKernel, {
    output: [layer.outputPlanes * outH * outW],
    constants: {
      inputPlanes: layer.inputPlanes,
      inH,
      inW,
      kH: layer.kernelHeight,
      kW: layer.kernelWidth,
      strideX: layer.strideX,
      strideY: layer.strideY,
      padX: layer.padX,
      padY: layer.padY,
      outW,
      planeSize: outH * outW,
      weightStride,
      alpha,
    },
    loopMaxIterations: Math.max(layer.inputPlanes, layer.kernelHeight, layer.kernelWidth),
    precision: "single",
  });
  kernels.set(key, kernel);
  return kernel;
}

export function gpuConvForward(
  input: Tensor,
  layer: ConvLayer,
  weights: Float32Array,
  bias: Float32Array,
  alpha: number,
): Tensor {
  if (input.channels !== layer.inputPlanes) {
    throw new Error(`conv expected ${layer.inputPlanes} input planes, received ${input.channels}`);
  }
  const outH = convOutputSize(input.height, layer.kernelHeight, layer.strideY, layer.padY);
  const outW = convOutputSize(input.width, layer.kernelWidth, layer.strideX, layer.padX);
  const kernel = kernelFor(layer, input.height, input.width, outH, outW, alpha);
  const result = kernel(input.data, weights, bias);
  const out = createTensor(layer.outputPlanes, outH, outW);
  out.data.set(result);
  return out;
}
