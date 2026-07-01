import { createTensor, fromData } from "../core/tensor.js";
import type { Tensor } from "../core/tensor.js";
import type { ConvLayer } from "../model/types.js";
import { convOutputSize } from "./conv.js";
import { getWasm, disableWasm } from "../wasm/loader.js";

export function cpuConvForward(
  input: Tensor,
  layer: ConvLayer,
  weights: Float32Array,
  bias: Float32Array,
  alpha: number,
): Tensor {
  if (input.channels !== layer.inputPlanes) {
    throw new Error(`conv expected ${layer.inputPlanes} input planes, received ${input.channels}`);
  }
  const { kernelWidth: kW, kernelHeight: kH, strideX, strideY, padX, padY, inputPlanes, outputPlanes } = layer;
  const inH = input.height;
  const inW = input.width;
  const outH = convOutputSize(inH, kH, strideY, padY);
  const outW = convOutputSize(inW, kW, strideX, padX);
  if (outH <= 0 || outW <= 0) {
    throw new Error(`conv kernel ${kW}x${kH} is larger than padded input ${inW}x${inH}`);
  }

  const wasm = getWasm();
  if (wasm) {
    try {
      const data = wasm.convForward(
        input.data,
        inH,
        inW,
        weights,
        bias,
        inputPlanes,
        outputPlanes,
        kW,
        kH,
        strideX,
        strideY,
        padX,
        padY,
        alpha,
        outH,
        outW,
      );
      return fromData(outputPlanes, outH, outW, data);
    } catch {
      disableWasm();
    }
  }

  const out = createTensor(outputPlanes, outH, outW);
  const outData = out.data;
  const inData = input.data;
  const weightStride = inputPlanes * kH * kW;
  const planeSize = outH * outW;

  for (let o = 0; o < outputPlanes; o++) {
    const biasValue = bias[o] as number;
    const wBase = o * weightStride;
    const outPlane = o * planeSize;
    for (let oy = 0; oy < outH; oy++) {
      const baseY = oy * strideY - padY;
      for (let ox = 0; ox < outW; ox++) {
        const baseX = ox * strideX - padX;
        let sum = biasValue;
        let w = wBase;
        for (let i = 0; i < inputPlanes; i++) {
          const planeBase = i * inH * inW;
          for (let ky = 0; ky < kH; ky++) {
            const iy = baseY + ky;
            const rowInRange = iy >= 0 && iy < inH;
            const rowBase = planeBase + iy * inW;
            for (let kx = 0; kx < kW; kx++) {
              const ix = baseX + kx;
              if (rowInRange && ix >= 0 && ix < inW) {
                sum += (inData[rowBase + ix] as number) * (weights[w] as number);
              }
              w++;
            }
          }
        }
        outData[outPlane + oy * outW + ox] = Math.max(sum, alpha * sum);
      }
    }
  }
  return out;
}
