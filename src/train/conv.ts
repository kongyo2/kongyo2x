import type { Tensor } from "../core/tensor.js";
import { createTensor } from "../core/tensor.js";

export interface ConvLayerParams {
  cin: number;
  cout: number;
  kh: number;
  kw: number;
  alpha: number;
  weights: Float32Array;
  bias: Float32Array;
  gradWeights: Float32Array;
  gradBias: Float32Array;
}

export function validOutputSize(inputSize: number, kernel: number): number {
  return inputSize - kernel + 1;
}

let scratchCol: Float32Array = new Float32Array(0);
let scratchDcol: Float32Array = new Float32Array(0);

function ensure(buffer: Float32Array, length: number): Float32Array {
  return buffer.length >= length ? buffer : new Float32Array(length);
}

function im2col(input: Tensor, kh: number, kw: number, outH: number, outW: number, k: number): Float32Array {
  const inW = input.width;
  const planeIn = input.height * inW;
  const inData = input.data;
  const cin = input.channels;
  const col = ensure(scratchCol, outH * outW * k);
  scratchCol = col;
  for (let oy = 0; oy < outH; oy++) {
    for (let ox = 0; ox < outW; ox++) {
      let dst = (oy * outW + ox) * k;
      for (let i = 0; i < cin; i++) {
        const inPlane = i * planeIn;
        for (let ky = 0; ky < kh; ky++) {
          const inRow = inPlane + (oy + ky) * inW + ox;
          for (let kx = 0; kx < kw; kx++) {
            col[dst++] = inData[inRow + kx] as number;
          }
        }
      }
    }
  }
  return col;
}

export function convForward(input: Tensor, layer: ConvLayerParams): Tensor {
  const { cin, cout, kh, kw } = layer;
  if (input.channels !== cin) {
    throw new Error(`conv expected ${cin} input planes, received ${input.channels}`);
  }
  const outH = validOutputSize(input.height, kh);
  const outW = validOutputSize(input.width, kw);
  if (outH <= 0 || outW <= 0) {
    throw new Error(`conv kernel ${kw}x${kh} is larger than input ${input.width}x${input.height}`);
  }
  const k = cin * kh * kw;
  const p = outH * outW;
  const col = im2col(input, kh, kw, outH, outW, k);
  const out = createTensor(cout, outH, outW);
  const outData = out.data;
  const weights = layer.weights;
  const bias = layer.bias;
  for (let o = 0; o < cout; o++) {
    const wBase = o * k;
    const outBase = o * p;
    const biasValue = bias[o] as number;
    for (let pi = 0; pi < p; pi++) {
      const colBase = pi * k;
      let acc = biasValue;
      for (let ki = 0; ki < k; ki++) {
        acc += (weights[wBase + ki] as number) * (col[colBase + ki] as number);
      }
      outData[outBase + pi] = acc;
    }
  }
  return out;
}

export function convBackward(
  input: Tensor,
  layer: ConvLayerParams,
  dPre: Tensor,
  computeGradInput: boolean,
): Tensor | null {
  const { cin, cout, kh, kw } = layer;
  const inH = input.height;
  const inW = input.width;
  const outH = dPre.height;
  const outW = dPre.width;
  const k = cin * kh * kw;
  const p = outH * outW;
  const col = im2col(input, kh, kw, outH, outW, k);
  const dPreData = dPre.data;
  const weights = layer.weights;
  const gradWeights = layer.gradWeights;
  const gradBias = layer.gradBias;

  for (let o = 0; o < cout; o++) {
    const wBase = o * k;
    const outBase = o * p;
    let biasAcc = 0;
    for (let pi = 0; pi < p; pi++) {
      const g = dPreData[outBase + pi] as number;
      biasAcc += g;
      if (g === 0) {
        continue;
      }
      const colBase = pi * k;
      for (let ki = 0; ki < k; ki++) {
        gradWeights[wBase + ki] = (gradWeights[wBase + ki] as number) + g * (col[colBase + ki] as number);
      }
    }
    gradBias[o] = (gradBias[o] as number) + biasAcc;
  }

  if (!computeGradInput) {
    return null;
  }

  const dcol = ensure(scratchDcol, p * k);
  scratchDcol = dcol;
  dcol.fill(0, 0, p * k);
  for (let o = 0; o < cout; o++) {
    const wBase = o * k;
    const outBase = o * p;
    for (let pi = 0; pi < p; pi++) {
      const g = dPreData[outBase + pi] as number;
      if (g === 0) {
        continue;
      }
      const colBase = pi * k;
      for (let ki = 0; ki < k; ki++) {
        dcol[colBase + ki] = (dcol[colBase + ki] as number) + g * (weights[wBase + ki] as number);
      }
    }
  }

  const dInput = createTensor(cin, inH, inW);
  const dInData = dInput.data;
  const planeIn = inH * inW;
  for (let oy = 0; oy < outH; oy++) {
    for (let ox = 0; ox < outW; ox++) {
      const colBase = (oy * outW + ox) * k;
      for (let i = 0; i < cin; i++) {
        const inPlane = i * planeIn;
        const kBase = colBase + i * kh * kw;
        for (let ky = 0; ky < kh; ky++) {
          const inRow = inPlane + (oy + ky) * inW + ox;
          const kRow = kBase + ky * kw;
          for (let kx = 0; kx < kw; kx++) {
            dInData[inRow + kx] = (dInData[inRow + kx] as number) + (dcol[kRow + kx] as number);
          }
        }
      }
    }
  }
  return dInput;
}
