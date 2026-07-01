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
  const kernelSize = kH * kW;
  const weightStride = inputPlanes * kernelSize;
  const planeSize = outH * outW;
  const inPlaneSize = inH * inW;

  // Kernel taps outside the input contribute nothing, so clip the tap ranges
  // up front instead of bounds-checking every tap. The surviving taps
  // accumulate in the original order, keeping results bit-identical.
  for (let o = 0; o < outputPlanes; o++) {
    const biasValue = bias[o] as number;
    const wBase = o * weightStride;
    const outPlane = o * planeSize;
    for (let oy = 0; oy < outH; oy++) {
      const baseY = oy * strideY - padY;
      const kyStart = baseY < 0 ? -baseY : 0;
      const kyEnd = kH < inH - baseY ? kH : inH - baseY;
      const outRow = outPlane + oy * outW;
      for (let ox = 0; ox < outW; ox++) {
        const baseX = ox * strideX - padX;
        const kxStart = baseX < 0 ? -baseX : 0;
        const kxEnd = kW < inW - baseX ? kW : inW - baseX;
        let sum = biasValue;
        for (let i = 0; i < inputPlanes; i++) {
          const planeBase = i * inPlaneSize + baseY * inW + baseX;
          const wPlane = wBase + i * kernelSize;
          for (let ky = kyStart; ky < kyEnd; ky++) {
            const rowBase = planeBase + ky * inW;
            const wRow = wPlane + ky * kW;
            for (let kx = kxStart; kx < kxEnd; kx++) {
              sum += (inData[rowBase + kx] as number) * (weights[wRow + kx] as number);
            }
          }
        }
        outData[outRow + ox] = Math.max(sum, alpha * sum);
      }
    }
  }
  return out;
}
