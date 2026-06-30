import type { Rng } from "./rng.js";

export function heInit(weights: Float32Array, fanIn: number, alpha: number, rng: Rng): void {
  const std = Math.sqrt(2 / (fanIn * (1 + alpha * alpha)));
  for (let i = 0; i < weights.length; i++) {
    weights[i] = rng.gaussian() * std;
  }
}

export function linearInit(weights: Float32Array, fanIn: number, rng: Rng): void {
  const std = Math.sqrt(1 / fanIn);
  for (let i = 0; i < weights.length; i++) {
    weights[i] = rng.gaussian() * std;
  }
}
