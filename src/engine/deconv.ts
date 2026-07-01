import { createTensor, fromData } from "../core/tensor.js";
import type { Tensor } from "../core/tensor.js";
import type { DeconvLayer } from "../model/types.js";
import { getWasm, disableWasm } from "../wasm/loader.js";

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

  const wasm = getWasm();
  if (wasm) {
    try {
      const data = wasm.deconvForward(
        input.data,
        inH,
        inW,
        layer.weights,
        layer.bias,
        layer.inputPlanes,
        layer.outputPlanes,
        kW,
        kH,
        strideX,
        strideY,
        padX,
        padY,
        adjX,
        adjY,
        outH,
        outW,
      );
      return fromData(layer.outputPlanes, outH, outW, data);
    } catch {
      disableWasm();
    }
  }

  const out = createTensor(layer.outputPlanes, outH, outW);
  const outData = out.data;
  const inData = input.data;
  const weights = layer.weights;
  const planeSize = outH * outW;

  for (let o = 0; o < layer.outputPlanes; o++) {
    const biasValue = layer.bias[o] as number;
    outData.fill(biasValue, o * planeSize, (o + 1) * planeSize);
  }

  const kernelSize = kH * kW;
  for (let i = 0; i < layer.inputPlanes; i++) {
    const inPlaneBase = i * inH * inW;
    const weightInBase = i * layer.outputPlanes * kernelSize;
    for (let iy = 0; iy < inH; iy++) {
      const oyBase = iy * strideY - padY;
      const kyStart = oyBase < 0 ? -oyBase : 0;
      const kyEnd = kH < outH - oyBase ? kH : outH - oyBase;
      for (let ix = 0; ix < inW; ix++) {
        const v = inData[inPlaneBase + iy * inW + ix] as number;
        if (v === 0) {
          continue;
        }
        const oxBase = ix * strideX - padX;
        const kxStart = oxBase < 0 ? -oxBase : 0;
        const kxEnd = kW < outW - oxBase ? kW : outW - oxBase;
        for (let o = 0; o < layer.outputPlanes; o++) {
          const weightBase = weightInBase + o * kernelSize;
          const outPlaneBase = o * planeSize + oyBase * outW + oxBase;
          for (let ky = kyStart; ky < kyEnd; ky++) {
            const outRowBase = outPlaneBase + ky * outW;
            const weightRowBase = weightBase + ky * kW;
            for (let kx = kxStart; kx < kxEnd; kx++) {
              const pos = outRowBase + kx;
              outData[pos] = (outData[pos] as number) + (weights[weightRowBase + kx] as number) * v;
            }
          }
        }
      }
    }
  }
  return out;
}
