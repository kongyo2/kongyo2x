import { readFileSync } from "node:fs";

interface WasmExports {
  memory: WebAssembly.Memory;
  kw_alloc(size: number): number;
  kw_dealloc(ptr: number, size: number): void;
  conv_forward(
    inPtr: number,
    inH: number,
    inW: number,
    wPtr: number,
    bPtr: number,
    outPtr: number,
    inputPlanes: number,
    outputPlanes: number,
    kw: number,
    kh: number,
    strideX: number,
    strideY: number,
    padX: number,
    padY: number,
    alpha: number,
  ): void;
  deconv_forward(
    inPtr: number,
    inH: number,
    inW: number,
    wPtr: number,
    bPtr: number,
    outPtr: number,
    inputPlanes: number,
    outputPlanes: number,
    kw: number,
    kh: number,
    strideX: number,
    strideY: number,
    padX: number,
    padY: number,
    adjX: number,
    adjY: number,
  ): void;
  conv_forward_train(
    inPtr: number,
    inH: number,
    inW: number,
    wPtr: number,
    bPtr: number,
    outPtr: number,
    cin: number,
    cout: number,
    kh: number,
    kw: number,
  ): void;
  conv_backward_train(
    inPtr: number,
    inH: number,
    inW: number,
    wPtr: number,
    dprePtr: number,
    outH: number,
    outW: number,
    gradWPtr: number,
    gradBPtr: number,
    dinPtr: number,
    cin: number,
    cout: number,
    kh: number,
    kw: number,
    computeGradInput: number,
  ): void;
  resize_lanczos(
    inPtr: number,
    channels: number,
    inH: number,
    inW: number,
    xIdxPtr: number,
    xWPtr: number,
    xTaps: number,
    outW: number,
    yIdxPtr: number,
    yWPtr: number,
    yTaps: number,
    outH: number,
    outPtr: number,
  ): void;
  make_border(rgbPtr: number, alphaPtr: number, outPtr: number, height: number, width: number, offset: number): void;
}

export interface WasmKernels {
  convForward(
    input: Float32Array,
    inH: number,
    inW: number,
    weights: Float32Array,
    bias: Float32Array,
    inputPlanes: number,
    outputPlanes: number,
    kW: number,
    kH: number,
    strideX: number,
    strideY: number,
    padX: number,
    padY: number,
    alpha: number,
    outH: number,
    outW: number,
  ): Float32Array;
  deconvForward(
    input: Float32Array,
    inH: number,
    inW: number,
    weights: Float32Array,
    bias: Float32Array,
    inputPlanes: number,
    outputPlanes: number,
    kW: number,
    kH: number,
    strideX: number,
    strideY: number,
    padX: number,
    padY: number,
    adjX: number,
    adjY: number,
    outH: number,
    outW: number,
  ): Float32Array;
  convForwardTrain(
    input: Float32Array,
    inH: number,
    inW: number,
    weights: Float32Array,
    bias: Float32Array,
    cin: number,
    cout: number,
    kh: number,
    kw: number,
    outH: number,
    outW: number,
  ): Float32Array;
  convBackwardTrain(
    input: Float32Array,
    inH: number,
    inW: number,
    weights: Float32Array,
    dPre: Float32Array,
    outH: number,
    outW: number,
    gradWeights: Float32Array,
    gradBias: Float32Array,
    cin: number,
    cout: number,
    kh: number,
    kw: number,
    computeGradInput: boolean,
  ): Float32Array | null;
  resizeLanczos(
    input: Float32Array,
    channels: number,
    inH: number,
    inW: number,
    xIdx: Int32Array,
    xWeights: Float32Array,
    xTaps: number,
    outW: number,
    yIdx: Int32Array,
    yWeights: Float32Array,
    yTaps: number,
    outH: number,
  ): Float32Array;
  makeBorder(rgb: Float32Array, alpha: Float32Array, height: number, width: number, offset: number): Float32Array;
}

const WASM_URL = new URL("../../wasm/kongyo2x.wasm", import.meta.url);

function isDisabled(): boolean {
  return process.env.KONGYO2X_DISABLE_WASM === "1" || process.env.KONGYO2X_WASM === "0";
}

function instantiate(): WasmExports | undefined {
  if (isDisabled()) {
    return undefined;
  }
  try {
    const bytes = readFileSync(WASM_URL);
    const module = new WebAssembly.Module(bytes);
    const instance = new WebAssembly.Instance(module, {});
    const exports = instance.exports as Record<string, unknown>;
    if (!(exports.memory instanceof WebAssembly.Memory)) {
      return undefined;
    }
    const required = [
      "kw_alloc",
      "kw_dealloc",
      "conv_forward",
      "deconv_forward",
      "conv_forward_train",
      "conv_backward_train",
      "resize_lanczos",
      "make_border",
    ];
    for (const name of required) {
      if (typeof exports[name] !== "function") {
        return undefined;
      }
    }
    return exports as unknown as WasmExports;
  } catch {
    return undefined;
  }
}

function build(ex: WasmExports): WasmKernels {
  const f32 = (ptr: number, len: number): Float32Array => new Float32Array(ex.memory.buffer, ptr, len);
  const i32 = (ptr: number, len: number): Int32Array => new Int32Array(ex.memory.buffer, ptr, len);
  const alloc = (elements: number): number => ex.kw_alloc(elements * 4);
  const free = (ptr: number, elements: number): void => ex.kw_dealloc(ptr, elements * 4);

  // Inference weights and biases never change once a model is loaded, so each
  // distinct Float32Array is uploaded to Wasm memory once and the buffer is
  // reused for every subsequent block and layer call; it is reclaimed when the
  // source array is garbage collected. Training kernels keep per-call uploads
  // because the optimizer rewrites their parameter arrays in place.
  const staticUploads = new WeakMap<Float32Array, number>();
  const reclaimer = new FinalizationRegistry<{ ptr: number; elements: number }>(({ ptr, elements }) => {
    free(ptr, elements);
  });
  const uploadStatic = (data: Float32Array): number => {
    let ptr = staticUploads.get(data);
    if (ptr === undefined) {
      ptr = alloc(data.length);
      f32(ptr, data.length).set(data);
      staticUploads.set(data, ptr);
      reclaimer.register(data, { ptr, elements: data.length });
    }
    return ptr;
  };

  return {
    convForward(input, inH, inW, weights, bias, ip, op, kW, kH, sx, sy, px, py, alpha, outH, outW) {
      const inLen = ip * inH * inW;
      const outLen = op * outH * outW;
      const wPtr = uploadStatic(weights);
      const bPtr = uploadStatic(bias);
      const inPtr = alloc(inLen);
      const outPtr = alloc(outLen);
      try {
        f32(inPtr, inLen).set(input.subarray(0, inLen));
        ex.conv_forward(inPtr, inH, inW, wPtr, bPtr, outPtr, ip, op, kW, kH, sx, sy, px, py, alpha);
        return Float32Array.from(f32(outPtr, outLen));
      } finally {
        free(inPtr, inLen);
        free(outPtr, outLen);
      }
    },

    deconvForward(input, inH, inW, weights, bias, ip, op, kW, kH, sx, sy, px, py, adjX, adjY, outH, outW) {
      const inLen = ip * inH * inW;
      const outLen = op * outH * outW;
      const wPtr = uploadStatic(weights);
      const bPtr = uploadStatic(bias);
      const inPtr = alloc(inLen);
      const outPtr = alloc(outLen);
      try {
        f32(inPtr, inLen).set(input.subarray(0, inLen));
        ex.deconv_forward(inPtr, inH, inW, wPtr, bPtr, outPtr, ip, op, kW, kH, sx, sy, px, py, adjX, adjY);
        return Float32Array.from(f32(outPtr, outLen));
      } finally {
        free(inPtr, inLen);
        free(outPtr, outLen);
      }
    },

    convForwardTrain(input, inH, inW, weights, bias, cin, cout, kh, kw, outH, outW) {
      const inLen = cin * inH * inW;
      const wLen = cout * cin * kh * kw;
      const outLen = cout * outH * outW;
      const inPtr = alloc(inLen);
      const wPtr = alloc(wLen);
      const bPtr = alloc(cout);
      const outPtr = alloc(outLen);
      try {
        f32(inPtr, inLen).set(input.subarray(0, inLen));
        f32(wPtr, wLen).set(weights.subarray(0, wLen));
        f32(bPtr, cout).set(bias.subarray(0, cout));
        ex.conv_forward_train(inPtr, inH, inW, wPtr, bPtr, outPtr, cin, cout, kh, kw);
        return Float32Array.from(f32(outPtr, outLen));
      } finally {
        free(inPtr, inLen);
        free(wPtr, wLen);
        free(bPtr, cout);
        free(outPtr, outLen);
      }
    },

    convBackwardTrain(input, inH, inW, weights, dPre, outH, outW, gradW, gradB, cin, cout, kh, kw, computeGradInput) {
      const inLen = cin * inH * inW;
      const wLen = cout * cin * kh * kw;
      const dpreLen = cout * outH * outW;
      const inPtr = alloc(inLen);
      const wPtr = alloc(wLen);
      const dprePtr = alloc(dpreLen);
      const gradWPtr = alloc(wLen);
      const gradBPtr = alloc(cout);
      const dinPtr = computeGradInput ? alloc(inLen) : 0;
      try {
        f32(inPtr, inLen).set(input.subarray(0, inLen));
        f32(wPtr, wLen).set(weights.subarray(0, wLen));
        f32(dprePtr, dpreLen).set(dPre.subarray(0, dpreLen));
        f32(gradWPtr, wLen).set(gradW.subarray(0, wLen));
        f32(gradBPtr, cout).set(gradB.subarray(0, cout));
        ex.conv_backward_train(
          inPtr,
          inH,
          inW,
          wPtr,
          dprePtr,
          outH,
          outW,
          gradWPtr,
          gradBPtr,
          dinPtr,
          cin,
          cout,
          kh,
          kw,
          computeGradInput ? 1 : 0,
        );
        gradW.set(f32(gradWPtr, wLen));
        gradB.set(f32(gradBPtr, cout));
        return computeGradInput ? Float32Array.from(f32(dinPtr, inLen)) : null;
      } finally {
        free(inPtr, inLen);
        free(wPtr, wLen);
        free(dprePtr, dpreLen);
        free(gradWPtr, wLen);
        free(gradBPtr, cout);
        if (dinPtr) {
          free(dinPtr, inLen);
        }
      }
    },

    resizeLanczos(input, channels, inH, inW, xIdx, xWeights, xTaps, outW, yIdx, yWeights, yTaps, outH) {
      const inLen = channels * inH * inW;
      const xLen = outW * xTaps;
      const yLen = outH * yTaps;
      const outLen = channels * outH * outW;
      const inPtr = alloc(inLen);
      const xIdxPtr = alloc(xLen);
      const xWPtr = alloc(xLen);
      const yIdxPtr = alloc(yLen);
      const yWPtr = alloc(yLen);
      const outPtr = alloc(outLen);
      try {
        f32(inPtr, inLen).set(input.subarray(0, inLen));
        i32(xIdxPtr, xLen).set(xIdx.subarray(0, xLen));
        f32(xWPtr, xLen).set(xWeights.subarray(0, xLen));
        i32(yIdxPtr, yLen).set(yIdx.subarray(0, yLen));
        f32(yWPtr, yLen).set(yWeights.subarray(0, yLen));
        ex.resize_lanczos(inPtr, channels, inH, inW, xIdxPtr, xWPtr, xTaps, outW, yIdxPtr, yWPtr, yTaps, outH, outPtr);
        return Float32Array.from(f32(outPtr, outLen));
      } finally {
        free(inPtr, inLen);
        free(xIdxPtr, xLen);
        free(xWPtr, xLen);
        free(yIdxPtr, yLen);
        free(yWPtr, yLen);
        free(outPtr, outLen);
      }
    },

    makeBorder(rgb, alpha, height, width, offset) {
      const size = height * width;
      const rgbLen = size * 3;
      const rgbPtr = alloc(rgbLen);
      const alphaPtr = alloc(size);
      const outPtr = alloc(rgbLen);
      try {
        f32(rgbPtr, rgbLen).set(rgb.subarray(0, rgbLen));
        f32(alphaPtr, size).set(alpha.subarray(0, size));
        ex.make_border(rgbPtr, alphaPtr, outPtr, height, width, offset);
        return Float32Array.from(f32(outPtr, rgbLen));
      } finally {
        free(rgbPtr, rgbLen);
        free(alphaPtr, size);
        free(outPtr, rgbLen);
      }
    },
  };
}

let resolved = false;
let kernels: WasmKernels | undefined;

export function getWasm(): WasmKernels | undefined {
  if (!resolved) {
    resolved = true;
    const ex = instantiate();
    kernels = ex ? build(ex) : undefined;
  }
  return kernels;
}

export function disableWasm(): void {
  resolved = true;
  kernels = undefined;
}

export function isWasmAvailable(): boolean {
  return getWasm() !== undefined;
}
