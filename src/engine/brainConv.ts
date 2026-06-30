import { createTensor } from "../core/tensor.js";
import type { Tensor } from "../core/tensor.js";
import type { ConvLayer } from "../model/types.js";
import type { BrainNeuralNetwork } from "../brain/loader.js";
import { convOutputSize } from "./conv.js";

export function brainConvForward(input: Tensor, layer: ConvLayer, net: BrainNeuralNetwork): Tensor {
  if (input.channels !== layer.inputPlanes) {
    throw new Error(`conv expected ${layer.inputPlanes} input planes, received ${input.channels}`);
  }
  const { kernelWidth: kW, kernelHeight: kH, strideX, strideY, padX, padY } = layer;
  const inH = input.height;
  const inW = input.width;
  const outH = convOutputSize(inH, kH, strideY, padY);
  const outW = convOutputSize(inW, kW, strideX, padX);
  const out = createTensor(layer.outputPlanes, outH, outW);
  const outData = out.data;
  const inData = input.data;
  const inSize = layer.inputPlanes * kH * kW;
  const patch = new Float32Array(inSize);
  const planeSize = outH * outW;

  for (let oy = 0; oy < outH; oy++) {
    const inYBase = oy * strideY - padY;
    for (let ox = 0; ox < outW; ox++) {
      const inXBase = ox * strideX - padX;
      let p = 0;
      for (let i = 0; i < layer.inputPlanes; i++) {
        const inPlaneBase = i * inH * inW;
        for (let ky = 0; ky < kH; ky++) {
          const iy = inYBase + ky;
          const inRowBase = inPlaneBase + iy * inW;
          const rowInRange = iy >= 0 && iy < inH;
          for (let kx = 0; kx < kW; kx++) {
            const ix = inXBase + kx;
            patch[p++] = rowInRange && ix >= 0 && ix < inW ? (inData[inRowBase + ix] as number) : 0;
          }
        }
      }
      const result = net.run(patch);
      const outPos = oy * outW + ox;
      for (let o = 0; o < layer.outputPlanes; o++) {
        outData[o * planeSize + outPos] = result[o] as number;
      }
    }
  }
  return out;
}
