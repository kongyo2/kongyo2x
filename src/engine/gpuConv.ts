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

type KernelFn = (this: KernelThis, ...inputs: Float32Array[]) => number;
type CompiledKernel = (...inputs: Float32Array[]) => ArrayLike<number>;

interface KernelSettings {
  output: [number];
  precision: "single";
  constants?: KernelThis["constants"];
  loopMaxIterations?: number;
}

interface GpuInstance {
  mode: string;
  createKernel(fn: KernelFn, settings: KernelSettings): CompiledKernel;
  destroy(): void;
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

function probeKernel(this: KernelThis, x: Float32Array): number {
  return (x[this.thread.x] as number) + 1;
}

interface GlContext {
  getExtension(name: string): { destroy(): void } | null;
}

type CreateGlContext = (width: number, height: number, options: Record<string, unknown>) => GlContext | null;

const nodeRequire = createRequire(import.meta.url);

function glUsable(): boolean {
  try {
    const createContext = nodeRequire("gl") as CreateGlContext;
    const context = createContext(1, 1, {});
    if (!context) {
      return false;
    }
    context.getExtension("STACKGL_destroy_context")?.destroy();
    return true;
  } catch {
    return false;
  }
}

function createGpu(): GpuInstance {
  const { GPU } = nodeRequire("gpu.js") as GpuModule;
  return new GPU();
}

function destroyGpu(instance: GpuInstance): void {
  try {
    instance.destroy();
  } catch {
    /* a half-initialized context may not destroy cleanly; nothing to recover */
  }
}

let gpu: GpuInstance | undefined;
let available: boolean | undefined;

function probeGpu(): boolean {
  if (process.env.KONGYO2X_DISABLE_GPU === "1" || process.env.KONGYO2X_GPU === "0") {
    return false;
  }
  if (!glUsable()) {
    return false;
  }
  let instance: GpuInstance | undefined;
  try {
    instance = createGpu();
    if (instance.mode === "cpu" || instance.mode === "dev") {
      destroyGpu(instance);
      return false;
    }
    const kernel = instance.createKernel(probeKernel, { output: [1], precision: "single" });
    const value = kernel(new Float32Array([1]))[0];
    if (typeof value !== "number" || Math.abs(value - 2) > 1e-3) {
      destroyGpu(instance);
      return false;
    }
    gpu = instance;
    return true;
  } catch {
    if (instance) {
      destroyGpu(instance);
    }
    return false;
  }
}

export function isGpuAvailable(): boolean {
  if (available === undefined) {
    available = probeGpu();
  }
  return available;
}

function getGpu(): GpuInstance {
  if (!gpu) {
    gpu = createGpu();
  }
  return gpu;
}

const kernels = new Map<string, CompiledKernel>();

function kernelFor(
  layer: ConvLayer,
  inH: number,
  inW: number,
  outH: number,
  outW: number,
  alpha: number,
): CompiledKernel {
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
