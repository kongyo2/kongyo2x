import { createTensor } from "../core/tensor.js";
import type { Tensor } from "../core/tensor.js";
import type { DeconvLayer } from "../model/types.js";

export function deconvOutputSize(inputSize: number, kernel: number, stride: number, pad: number, adj: number): number {
  return (inputSize - 1) * stride - 2 * pad + kernel + adj;
}

export function spatialFullConvolution(input: Tensor, layer: DeconvLayer): Tensor {
  if (input.channels !== layer.inputPlanes) {
    throw new Error(`deconv expected ${layer.inputPlanes} input planes, received ${input.channels}`);
  }
  const { kernelWidth: kW, kernelHeight: kH, strideX, strideY, padX, padY, adjX, adjY } = layer;
  const inH = input.height;
  const inW = input.width;
  const outH = deconvOutputSize(inH, kH, strideY, padY, adjY);
  const outW = deconvOutputSize(inW, kW, strideX, padX, adjX);
  const out = createTensor(layer.outputPlanes, outH, outW);
  const outData = out.data;
  const inData = input.data;
  const weights = layer.weights;
  const planeSize = outH * outW;

  for (let o = 0; o < layer.outputPlanes; o++) {
    const biasValue = layer.bias[o] as number;
    outData.fill(biasValue, o * planeSize, (o + 1) * planeSize);
  }

  for (let i = 0; i < layer.inputPlanes; i++) {
    const inPlaneBase = i * inH * inW;
    const weightInBase = i * layer.outputPlanes * kH * kW;
    for (let iy = 0; iy < inH; iy++) {
      for (let ix = 0; ix < inW; ix++) {
        const v = inData[inPlaneBase + iy * inW + ix] as number;
        if (v === 0) {
          continue;
        }
        const oyBase = iy * strideY - padY;
        const oxBase = ix * strideX - padX;
        for (let o = 0; o < layer.outputPlanes; o++) {
          const weightBase = weightInBase + o * kH * kW;
          const outPlaneBase = o * planeSize;
          for (let ky = 0; ky < kH; ky++) {
            const oy = oyBase + ky;
            if (oy < 0 || oy >= outH) {
              continue;
            }
            const outRowBase = outPlaneBase + oy * outW;
            const weightRowBase = weightBase + ky * kW;
            for (let kx = 0; kx < kW; kx++) {
              const ox = oxBase + kx;
              if (ox >= 0 && ox < outW) {
                const pos = outRowBase + ox;
                outData[pos] = (outData[pos] as number) + (weights[weightRowBase + kx] as number) * v;
              }
            }
          }
        }
      }
    }
  }
  return out;
}
