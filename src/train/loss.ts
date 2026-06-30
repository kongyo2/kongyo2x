import type { Tensor } from "../core/tensor.js";
import { createTensor } from "../core/tensor.js";

export type LossKind = "charbonnier" | "mse";

export interface LossResult {
  loss: number;
  grad: Tensor;
}

export function computeLoss(output: Tensor, target: Tensor, kind: LossKind, epsilon = 1e-3): LossResult {
  if (output.data.length !== target.data.length) {
    throw new Error(`loss shape mismatch: ${output.data.length} vs ${target.data.length}`);
  }
  const n = output.data.length;
  const grad = createTensor(output.channels, output.height, output.width);
  const out = output.data;
  const tgt = target.data;
  const g = grad.data;
  let loss = 0;
  if (kind === "mse") {
    for (let i = 0; i < n; i++) {
      const d = (out[i] as number) - (tgt[i] as number);
      loss += d * d;
      g[i] = (2 * d) / n;
    }
    return { loss: loss / n, grad };
  }
  const eps2 = epsilon * epsilon;
  for (let i = 0; i < n; i++) {
    const d = (out[i] as number) - (tgt[i] as number);
    const root = Math.sqrt(d * d + eps2);
    loss += root;
    g[i] = d / root / n;
  }
  return { loss: loss / n, grad };
}

export function meanSquaredError(output: Tensor, target: Tensor): number {
  const n = output.data.length;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const d = (output.data[i] as number) - (target.data[i] as number);
    sum += d * d;
  }
  return sum / n;
}

export function psnr(mse: number): number {
  if (mse <= 1e-12) {
    return 100;
  }
  return 10 * Math.log10(1 / mse);
}
