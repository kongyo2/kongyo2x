import type { Tensor } from "../core/tensor.js";
import { createTensor } from "../core/tensor.js";

export function leakyRelu(pre: Tensor, alpha: number): Tensor {
  const out = createTensor(pre.channels, pre.height, pre.width);
  const src = pre.data;
  const dst = out.data;
  for (let i = 0; i < src.length; i++) {
    const v = src[i] as number;
    dst[i] = v > 0 ? v : alpha * v;
  }
  return out;
}

export function leakyReluGradInto(dst: Tensor, pre: Tensor, dAct: Tensor, alpha: number): void {
  const preData = pre.data;
  const dActData = dAct.data;
  const dstData = dst.data;
  for (let i = 0; i < preData.length; i++) {
    const slope = (preData[i] as number) > 0 ? 1 : alpha;
    dstData[i] = (dActData[i] as number) * slope;
  }
}
