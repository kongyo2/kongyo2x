import { createTensor } from "../core/tensor.js";
import type { Tensor } from "../core/tensor.js";

export function padEdge(input: Tensor, left: number, right: number, top: number, bottom: number): Tensor {
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
      for (let ox = 0; ox < outW; ox++) {
        let sx = ox - left;
        if (sx < 0) {
          sx = 0;
        } else if (sx >= inW) {
          sx = inW - 1;
        }
        dst[dstRow + ox] = src[srcRow + sx] as number;
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
      const dstRow = dstPlane + y * outW;
      for (let x = 0; x < outW; x++) {
        dst[dstRow + x] = src[srcRow + x] as number;
      }
    }
  }
  return out;
}
