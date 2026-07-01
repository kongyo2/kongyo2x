import { cloneTensor, createTensor, fromData } from "../core/tensor.js";
import type { Tensor } from "../core/tensor.js";
import { getWasm } from "../wasm/loader.js";

function box3x3Sum(plane: Float32Array, height: number, width: number): Float32Array {
  const out = new Float32Array(height * width);
  for (let y = 0; y < height; y++) {
    const y0 = y > 0 ? y - 1 : 0;
    const y1 = y < height - 1 ? y + 1 : height - 1;
    for (let x = 0; x < width; x++) {
      const x0 = x > 0 ? x - 1 : 0;
      const x1 = x < width - 1 ? x + 1 : width - 1;
      let sum = 0;
      for (let yy = y0; yy <= y1; yy++) {
        const rowBase = yy * width;
        for (let xx = x0; xx <= x1; xx++) {
          sum += plane[rowBase + xx] as number;
        }
      }
      out[y * width + x] = sum;
    }
  }
  return out;
}

export function makeBorder(rgb: Tensor, alpha: Tensor, offset: number): Tensor {
  if (rgb.channels !== 3) {
    throw new Error("makeBorder expects a 3-channel RGB tensor");
  }
  const { height, width } = rgb;

  const wasm = getWasm();
  if (wasm) {
    const data = wasm.makeBorder(rgb.data, alpha.data, height, width, offset);
    return fromData(3, height, width, data);
  }

  const size = height * width;
  const eps = 1e-7;
  const result = cloneTensor(rgb);
  const out = result.data;

  let mask = new Float32Array(size);
  for (let p = 0; p < size; p++) {
    mask[p] = (alpha.data[p] as number) > 0 ? 1 : 0;
  }
  for (let p = 0; p < size; p++) {
    if (mask[p] === 0) {
      out[p] = 0;
      out[size + p] = 0;
      out[2 * size + p] = 0;
    }
  }

  for (let step = 0; step < offset; step++) {
    const maskWeight = box3x3Sum(mask, height, width);
    for (let ch = 0; ch < 3; ch++) {
      const channelBase = ch * size;
      const blurred = box3x3Sum(out.subarray(channelBase, channelBase + size) as Float32Array, height, width);
      for (let p = 0; p < size; p++) {
        if (mask[p] === 0) {
          out[channelBase + p] = (blurred[p] as number) / ((maskWeight[p] as number) + eps);
        }
      }
    }
    const nextMask = new Float32Array(size);
    for (let p = 0; p < size; p++) {
      nextMask[p] = (maskWeight[p] as number) > 0 ? 1 : 0;
    }
    mask = nextMask;
  }

  for (let i = 0; i < out.length; i++) {
    const v = out[i] as number;
    out[i] = v < 0 ? 0 : v > 1 ? 1 : v;
  }
  return result;
}

export function ensureAlphaPlane(alpha: Tensor): Tensor {
  if (alpha.channels !== 1) {
    const plane = createTensor(1, alpha.height, alpha.width);
    plane.data.set(alpha.data.subarray(0, alpha.height * alpha.width));
    return plane;
  }
  return alpha;
}
