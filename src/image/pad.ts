import { createTensor } from "../core/tensor.js";
import type { Tensor } from "../core/tensor.js";

export function padEdge(input: Tensor, left: number, right: number, top: number, bottom: number): Tensor {
  if (left < 0 || right < 0 || top < 0 || bottom < 0) {
    throw new Error(`invalid padding ${left},${right},${top},${bottom} (expected non-negative values)`);
  }
  const inH = input.height;
  const inW = input.width;
  const outH = inH + top + bottom;
  const outW = inW + left + right;
  const out = createTensor(input.channels, outH, outW);
  const src = input.data;
  const dst = out.data;
  for (let c = 0; c < input.channels; c++) {
    const srcPlane = c * inH * inW;
    const dstPlane = c * outH * outW;
    for (let oy = 0; oy < outH; oy++) {
      let sy = oy - top;
      if (sy < 0) {
        sy = 0;
      } else if (sy >= inH) {
        sy = inH - 1;
      }
      const srcRow = srcPlane + sy * inW;
      const dstRow = dstPlane + oy * outW;
      if (left > 0) {
        dst.fill(src[srcRow] as number, dstRow, dstRow + left);
      }
      dst.set(src.subarray(srcRow, srcRow + inW), dstRow + left);
      if (right > 0) {
        dst.fill(src[srcRow + inW - 1] as number, dstRow + left + inW, dstRow + outW);
      }
    }
  }
  return out;
}

export function crop(input: Tensor, x1: number, y1: number, x2: number, y2: number): Tensor {
  const outW = x2 - x1;
  const outH = y2 - y1;
  if (outW <= 0 || outH <= 0) {
    throw new Error(`invalid crop region ${x1},${y1} -> ${x2},${y2}`);
  }
  if (x1 < 0 || y1 < 0 || x2 > input.width || y2 > input.height) {
    throw new Error(`crop region ${x1},${y1} -> ${x2},${y2} exceeds input ${input.width}x${input.height}`);
  }
  const out = createTensor(input.channels, outH, outW);
  const src = input.data;
  const dst = out.data;
  const inH = input.height;
  const inW = input.width;
  for (let c = 0; c < input.channels; c++) {
    const srcPlane = c * inH * inW;
    const dstPlane = c * outH * outW;
    for (let y = 0; y < outH; y++) {
      const srcRow = srcPlane + (y1 + y) * inW + x1;
      dst.set(src.subarray(srcRow, srcRow + outW), dstPlane + y * outW);
    }
  }
  return out;
}
