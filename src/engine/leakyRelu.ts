import type { Tensor } from "../core/tensor.js";

export function applyLeakyReluInPlace(t: Tensor, alpha: number): Tensor {
  const data = t.data;
  for (let i = 0; i < data.length; i++) {
    const v = data[i] as number;
    if (v < 0) {
      data[i] = v * alpha;
    }
  }
  return t;
}
