import type { Tensor } from "../core/tensor.js";
import { createTensor } from "../core/tensor.js";

export interface DeconvLayerParams {
  cin: number;
  cout: number;
  kh: number;
  kw: number;
  strideX: number;
  strideY: number;
  padX: number;
  padY: number;
  adjX: number;
  adjY: number;
  /** Weights indexed as weight[inPlane][outPlane][ky][kx], matching the inference engine. */
  weights: Float32Array;
  bias: Float32Array;
  gradWeights: Float32Array;
  gradBias: Float32Array;
}

export function deconvOutputSize(inputSize: number, kernel: number, stride: number, pad: number, adj: number): number {
  return (inputSize - 1) * stride - 2 * pad + kernel + adj;
}

export function deconvForward(input: Tensor, layer: DeconvLayerParams): Tensor {
  const { cin, cout, kh, kw, strideX, strideY, padX, padY, adjX, adjY } = layer;
  if (input.channels !== cin) {
    throw new Error(`deconv expected ${cin} input planes, received ${input.channels}`);
  }
  const inH = input.height;
  const inW = input.width;
  const outH = deconvOutputSize(inH, kh, strideY, padY, adjY);
  const outW = deconvOutputSize(inW, kw, strideX, padX, adjX);
  if (outH <= 0 || outW <= 0) {
    throw new Error(`deconv produced empty output ${outW}x${outH}`);
  }

  const out = createTensor(cout, outH, outW);
  const outData = out.data;
  const planeOut = outH * outW;
  for (let o = 0; o < cout; o++) {
    outData.fill(layer.bias[o] as number, o * planeOut, (o + 1) * planeOut);
  }

  const inData = input.data;
  const weights = layer.weights;
  const planeIn = inH * inW;
  for (let i = 0; i < cin; i++) {
    const inPlane = i * planeIn;
    const weightInBase = i * cout * kh * kw;
    for (let iy = 0; iy < inH; iy++) {
      const oyBase = iy * strideY - padY;
      for (let ix = 0; ix < inW; ix++) {
        const v = inData[inPlane + iy * inW + ix] as number;
        if (v === 0) {
          continue;
        }
        const oxBase = ix * strideX - padX;
        for (let o = 0; o < cout; o++) {
          const weightBase = weightInBase + o * kh * kw;
          const outPlane = o * planeOut;
          for (let ky = 0; ky < kh; ky++) {
            const oy = oyBase + ky;
            if (oy < 0 || oy >= outH) {
              continue;
            }
            const outRow = outPlane + oy * outW;
            const weightRow = weightBase + ky * kw;
            for (let kx = 0; kx < kw; kx++) {
              const ox = oxBase + kx;
              if (ox >= 0 && ox < outW) {
                outData[outRow + ox] = (outData[outRow + ox] as number) + (weights[weightRow + kx] as number) * v;
              }
            }
          }
        }
      }
    }
  }
  return out;
}

export function deconvBackward(
  input: Tensor,
  layer: DeconvLayerParams,
  dOut: Tensor,
  computeGradInput: boolean,
): Tensor | null {
  const { cin, cout, kh, kw, strideX, strideY, padX, padY, adjX, adjY } = layer;
  const inH = input.height;
  const inW = input.width;
  if (input.channels !== cin) {
    throw new Error(`deconv backward expected ${cin} input planes, received ${input.channels}`);
  }
  if (dOut.channels !== cout) {
    throw new Error(`deconv backward expected ${cout} gradient planes, received ${dOut.channels}`);
  }
  const outH = dOut.height;
  const outW = dOut.width;
  const expectedH = deconvOutputSize(inH, kh, strideY, padY, adjY);
  const expectedW = deconvOutputSize(inW, kw, strideX, padX, adjX);
  if (outH !== expectedH || outW !== expectedW) {
    throw new Error(`deconv backward gradient is ${outW}x${outH}, expected ${expectedW}x${expectedH}`);
  }

  const inData = input.data;
  const dOutData = dOut.data;
  const weights = layer.weights;
  const gradWeights = layer.gradWeights;
  const gradBias = layer.gradBias;
  const planeIn = inH * inW;
  const planeOut = outH * outW;

  for (let o = 0; o < cout; o++) {
    let biasAcc = 0;
    const outPlane = o * planeOut;
    for (let pi = 0; pi < planeOut; pi++) {
      biasAcc += dOutData[outPlane + pi] as number;
    }
    gradBias[o] = (gradBias[o] as number) + biasAcc;
  }

  const dInput = computeGradInput ? createTensor(cin, inH, inW) : null;
  const dInData = dInput?.data;

  for (let i = 0; i < cin; i++) {
    const inPlane = i * planeIn;
    const weightInBase = i * cout * kh * kw;
    for (let iy = 0; iy < inH; iy++) {
      const oyBase = iy * strideY - padY;
      for (let ix = 0; ix < inW; ix++) {
        const inPos = inPlane + iy * inW + ix;
        const v = inData[inPos] as number;
        const oxBase = ix * strideX - padX;
        let inputAcc = 0;
        for (let o = 0; o < cout; o++) {
          const weightBase = weightInBase + o * kh * kw;
          const outPlane = o * planeOut;
          for (let ky = 0; ky < kh; ky++) {
            const oy = oyBase + ky;
            if (oy < 0 || oy >= outH) {
              continue;
            }
            const outRow = outPlane + oy * outW;
            const weightRow = weightBase + ky * kw;
            for (let kx = 0; kx < kw; kx++) {
              const ox = oxBase + kx;
              if (ox < 0 || ox >= outW) {
                continue;
              }
              const g = dOutData[outRow + ox] as number;
              gradWeights[weightRow + kx] = (gradWeights[weightRow + kx] as number) + v * g;
              if (dInData) {
                inputAcc += (weights[weightRow + kx] as number) * g;
              }
            }
          }
        }
        if (dInData) {
          dInData[inPos] = inputAcc;
        }
      }
    }
  }
  return dInput;
}
