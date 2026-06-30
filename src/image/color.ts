import { createTensor } from "../core/tensor.js";
import type { Tensor } from "../core/tensor.js";

function requireRgb(t: Tensor): void {
  if (t.channels !== 3) {
    throw new Error(`expected a 3-channel RGB tensor, received ${t.channels} channels`);
  }
}

export function rgb2yuv(rgb: Tensor): Tensor {
  requireRgb(rgb);
  const { height, width } = rgb;
  const size = height * width;
  const out = createTensor(3, height, width);
  const src = rgb.data;
  const dst = out.data;
  for (let p = 0; p < size; p++) {
    const r = src[p] as number;
    const g = src[size + p] as number;
    const b = src[2 * size + p] as number;
    dst[p] = 0.299 * r + 0.587 * g + 0.114 * b;
    dst[size + p] = -0.14713 * r - 0.28886 * g + 0.436 * b;
    dst[2 * size + p] = 0.615 * r - 0.51499 * g - 0.10001 * b;
  }
  return out;
}

export function yuv2rgb(yuv: Tensor): Tensor {
  requireRgb(yuv);
  const { height, width } = yuv;
  const size = height * width;
  const out = createTensor(3, height, width);
  const src = yuv.data;
  const dst = out.data;
  for (let p = 0; p < size; p++) {
    const y = src[p] as number;
    const u = src[size + p] as number;
    const v = src[2 * size + p] as number;
    dst[p] = y + 1.13983 * v;
    dst[size + p] = y - 0.39465 * u - 0.5806 * v;
    dst[2 * size + p] = y + 2.03211 * u;
  }
  return out;
}

export function rgb2y(rgb: Tensor): Tensor {
  requireRgb(rgb);
  const { height, width } = rgb;
  const size = height * width;
  const out = createTensor(1, height, width);
  const src = rgb.data;
  const dst = out.data;
  for (let p = 0; p < size; p++) {
    const r = src[p] as number;
    const g = src[size + p] as number;
    const b = src[2 * size + p] as number;
    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    y = y < 0 ? 0 : y > 1 ? 1 : y;
    dst[p] = y;
  }
  return out;
}
