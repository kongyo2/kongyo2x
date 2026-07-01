import { createTensor, fromData } from "../core/tensor.js";
import type { Tensor } from "../core/tensor.js";
import { getWasm } from "../wasm/loader.js";

export function resizeNearest(input: Tensor, outW: number, outH: number): Tensor {
  const inH = input.height;
  const inW = input.width;
  const out = createTensor(input.channels, outH, outW);
  const src = input.data;
  const dst = out.data;
  const xMap = new Int32Array(outW);
  for (let ox = 0; ox < outW; ox++) {
    let sx = Math.floor(((ox + 0.5) * inW) / outW);
    if (sx < 0) {
      sx = 0;
    } else if (sx >= inW) {
      sx = inW - 1;
    }
    xMap[ox] = sx;
  }
  for (let c = 0; c < input.channels; c++) {
    const srcPlane = c * inH * inW;
    const dstPlane = c * outH * outW;
    for (let oy = 0; oy < outH; oy++) {
      let sy = Math.floor(((oy + 0.5) * inH) / outH);
      if (sy < 0) {
        sy = 0;
      } else if (sy >= inH) {
        sy = inH - 1;
      }
      const srcRow = srcPlane + sy * inW;
      const dstRow = dstPlane + oy * outW;
      for (let ox = 0; ox < outW; ox++) {
        dst[dstRow + ox] = src[srcRow + (xMap[ox] as number)] as number;
      }
    }
  }
  return out;
}

interface ResampleTap {
  readonly indices: Int32Array;
  readonly weights: Float32Array;
  readonly taps: number;
}

function sinc(x: number): number {
  if (x === 0) {
    return 1;
  }
  const px = Math.PI * x;
  return Math.sin(px) / px;
}

function lanczosWeight(x: number, a: number): number {
  if (x <= -a || x >= a) {
    return 0;
  }
  return sinc(x) * sinc(x / a);
}

function buildLanczosTaps(inSize: number, outSize: number, a: number): ResampleTap {
  const scale = inSize / outSize;
  const support = scale > 1 ? a * scale : a;
  const tapsPerOutput = Math.ceil(support * 2) + 1;
  const indices = new Int32Array(outSize * tapsPerOutput);
  const weights = new Float32Array(outSize * tapsPerOutput);
  const filterScale = scale > 1 ? 1 / scale : 1;
  for (let o = 0; o < outSize; o++) {
    const center = (o + 0.5) * scale - 0.5;
    const start = Math.ceil(center - support);
    let sum = 0;
    const base = o * tapsPerOutput;
    for (let t = 0; t < tapsPerOutput; t++) {
      const tap = start + t;
      const w = lanczosWeight((tap - center) * filterScale, a);
      let clamped = tap;
      if (clamped < 0) {
        clamped = 0;
      } else if (clamped >= inSize) {
        clamped = inSize - 1;
      }
      indices[base + t] = clamped;
      weights[base + t] = w;
      sum += w;
    }
    if (sum !== 0) {
      for (let t = 0; t < tapsPerOutput; t++) {
        weights[base + t] = (weights[base + t] as number) / sum;
      }
    }
  }
  return { indices, weights, taps: tapsPerOutput };
}

export function resizeLanczos(input: Tensor, outW: number, outH: number, a = 3): Tensor {
  const inH = input.height;
  const inW = input.width;
  const xTap = buildLanczosTaps(inW, outW, a);
  const yTap = buildLanczosTaps(inH, outH, a);

  const wasm = getWasm();
  if (wasm) {
    const data = wasm.resizeLanczos(
      input.data,
      input.channels,
      inH,
      inW,
      xTap.indices,
      xTap.weights,
      xTap.taps,
      outW,
      yTap.indices,
      yTap.weights,
      yTap.taps,
      outH,
    );
    return fromData(input.channels, outH, outW, data);
  }

  const horizontal = createTensor(input.channels, inH, outW);
  {
    const src = input.data;
    const dst = horizontal.data;
    for (let c = 0; c < input.channels; c++) {
      const srcPlane = c * inH * inW;
      const dstPlane = c * inH * outW;
      for (let y = 0; y < inH; y++) {
        const srcRow = srcPlane + y * inW;
        const dstRow = dstPlane + y * outW;
        for (let ox = 0; ox < outW; ox++) {
          const base = ox * xTap.taps;
          let acc = 0;
          for (let t = 0; t < xTap.taps; t++) {
            acc += (xTap.weights[base + t] as number) * (src[srcRow + (xTap.indices[base + t] as number)] as number);
          }
          dst[dstRow + ox] = acc;
        }
      }
    }
  }

  const out = createTensor(input.channels, outH, outW);
  {
    const src = horizontal.data;
    const dst = out.data;
    for (let c = 0; c < input.channels; c++) {
      const srcPlane = c * inH * outW;
      const dstPlane = c * outH * outW;
      for (let oy = 0; oy < outH; oy++) {
        const base = oy * yTap.taps;
        const dstRow = dstPlane + oy * outW;
        for (let ox = 0; ox < outW; ox++) {
          let acc = 0;
          for (let t = 0; t < yTap.taps; t++) {
            const sy = yTap.indices[base + t] as number;
            acc += (yTap.weights[base + t] as number) * (src[srcPlane + sy * outW + ox] as number);
          }
          dst[dstRow + ox] = acc;
        }
      }
    }
  }
  return out;
}
