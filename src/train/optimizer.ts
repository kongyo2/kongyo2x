import type { Parameter } from "./network.js";

export interface AdamOptions {
  beta1: number;
  beta2: number;
  epsilon: number;
  weightDecay: number;
}

export const DEFAULT_ADAM: AdamOptions = {
  beta1: 0.9,
  beta2: 0.999,
  epsilon: 1e-8,
  weightDecay: 0,
};

export class Adam {
  private readonly params: Parameter[];
  private readonly options: AdamOptions;
  private readonly m: Float32Array[];
  private readonly v: Float32Array[];
  private step = 0;

  constructor(params: Parameter[], options: Partial<AdamOptions> = {}) {
    this.params = params;
    this.options = { ...DEFAULT_ADAM, ...options };
    this.m = params.map((p) => new Float32Array(p.value.length));
    this.v = params.map((p) => new Float32Array(p.value.length));
  }

  update(learningRate: number): void {
    this.step += 1;
    const { beta1, beta2, epsilon, weightDecay } = this.options;
    const biasCorrection1 = 1 - Math.pow(beta1, this.step);
    const biasCorrection2 = 1 - Math.pow(beta2, this.step);
    for (let pi = 0; pi < this.params.length; pi++) {
      const param = this.params[pi] as Parameter;
      const value = param.value;
      const grad = param.grad;
      const m = this.m[pi] as Float32Array;
      const v = this.v[pi] as Float32Array;
      for (let i = 0; i < value.length; i++) {
        const g = grad[i] as number;
        const mi = beta1 * (m[i] as number) + (1 - beta1) * g;
        const vi = beta2 * (v[i] as number) + (1 - beta2) * g * g;
        m[i] = mi;
        v[i] = vi;
        const mHat = mi / biasCorrection1;
        const vHat = vi / biasCorrection2;
        let next = (value[i] as number) - learningRate * (mHat / (Math.sqrt(vHat) + epsilon));
        if (weightDecay !== 0) {
          next -= learningRate * weightDecay * (value[i] as number);
        }
        value[i] = next;
      }
    }
  }
}

export interface ScheduleOptions {
  baseLr: number;
  minLr: number;
  warmup: number;
  totalSteps: number;
}

export function cosineLr(step: number, options: ScheduleOptions): number {
  const { baseLr, minLr, warmup, totalSteps } = options;
  if (step < warmup) {
    return (baseLr * (step + 1)) / Math.max(1, warmup);
  }
  const progress = Math.min(1, (step - warmup) / Math.max(1, totalSteps - warmup));
  const cosine = 0.5 * (1 + Math.cos(Math.PI * progress));
  return minLr + (baseLr - minLr) * cosine;
}

export function clipGradNorm(params: Parameter[], maxNorm: number): number {
  let sumSquares = 0;
  for (const param of params) {
    const grad = param.grad;
    for (let i = 0; i < grad.length; i++) {
      const g = grad[i] as number;
      sumSquares += g * g;
    }
  }
  const norm = Math.sqrt(sumSquares);
  if (maxNorm > 0 && norm > maxNorm) {
    const scale = maxNorm / (norm + 1e-12);
    for (const param of params) {
      const grad = param.grad;
      for (let i = 0; i < grad.length; i++) {
        grad[i] = (grad[i] as number) * scale;
      }
    }
  }
  return norm;
}

export class Ema {
  private readonly shadow: Float32Array[];
  private readonly decay: number;
  private steps = 0;

  constructor(params: Parameter[], decay: number) {
    this.decay = decay;
    this.shadow = params.map((p) => Float32Array.from(p.value));
  }

  get enabled(): boolean {
    return this.decay > 0;
  }

  update(params: Parameter[]): void {
    if (!this.enabled) {
      return;
    }
    this.steps += 1;
    const d = Math.min(this.decay, (1 + this.steps) / (10 + this.steps));
    for (let pi = 0; pi < params.length; pi++) {
      const value = (params[pi] as Parameter).value;
      const shadow = this.shadow[pi] as Float32Array;
      for (let i = 0; i < value.length; i++) {
        shadow[i] = d * (shadow[i] as number) + (1 - d) * (value[i] as number);
      }
    }
  }

  writeInto(params: Parameter[]): void {
    if (!this.enabled) {
      return;
    }
    for (let pi = 0; pi < params.length; pi++) {
      (params[pi] as Parameter).value.set(this.shadow[pi] as Float32Array);
    }
  }
}
