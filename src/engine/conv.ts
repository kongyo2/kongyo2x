import { createTensor } from "../core/tensor.js";
import type { Tensor } from "../core/tensor.js";
import type { ConvLayer } from "../model/types.js";

export function convOutputSize(inputSize: number, kernel: number, stride: number, pad: number): number {
  return Math.floor((inputSize + 2 * pad - kernel) / stride) + 1;
}

export function spatialConvolutionMM(input: Tensor, layer: ConvLayer): Tensor {
  if (input.channels !== layer.inputPlanes) {
    throw new Error(`conv expected ${layer.inputPlanes} input planes, received ${input.channels}`);
  }
  const { kernelWidth: kW, kernelHeight: kH, strideX, strideY, padX, padY } = layer;
  const outH = convOutputSize(input.height, kH, strideY, padY);
  const outW = convOutputSize(input.width, kW, strideX, padX);
  const out = createTensor(layer.outputPlanes, outH, outW);
  const inH = input.height;
  const inW = input.width;
  const inData = input.data;
  const outData = out.data;
  const weights = layer.weights;
  const rowLength = layer.inputPlanes * kH * kW;

  for (let o = 0; o < layer.outputPlanes; o++) {
    const weightBase = o * rowLength;
    const biasValue = layer.bias[o] as number;
    const outPlaneBase = o * outH * outW;
    for (let oy = 0; oy < outH; oy++) {
      const inYBase = oy * strideY - padY;
      for (let ox = 0; ox < outW; ox++) {
        const inXBase = ox * strideX - padX;
        let sum = biasValue;
        let w = weightBase;
        for (let i = 0; i < layer.inputPlanes; i++) {
          const inPlaneBase = i * inH * inW;
          for (let ky = 0; ky < kH; ky++) {
            const iy = inYBase + ky;
            if (iy < 0 || iy >= inH) {
              w += kW;
              continue;
            }
            const inRowBase = inPlaneBase + iy * inW;
            for (let kx = 0; kx < kW; kx++) {
              const ix = inXBase + kx;
              if (ix >= 0 && ix < inW) {
                sum += (weights[w] as number) * (inData[inRowBase + ix] as number);
              }
              w++;
            }
          }
        }
        outData[outPlaneBase + oy * outW + ox] = sum;
      }
    }
  }
  return out;
}
