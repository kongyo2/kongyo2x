import type { Tensor } from "../core/tensor.js";
import type { Kongyo2xModel } from "../model/model.js";
import type { ModelLayer } from "../model/types.js";
import { spatialFullConvolution } from "./deconv.js";
import { brainConvForward } from "./brainConv.js";
import { gpuConvForward, isGpuAvailable } from "./gpuConv.js";

export function runModel(model: Kongyo2xModel, input: Tensor): Tensor {
  const useGpu = isGpuAvailable();
  let current = input;
  for (let index = 0; index < model.layers.length; index++) {
    const layer = model.layers[index] as ModelLayer;
    if (layer.kind === "deconv") {
      current = spatialFullConvolution(current, layer);
      continue;
    }
    if (useGpu) {
      const params = model.convParams(index);
      if (params) {
        current = gpuConvForward(current, layer, params.weights, params.bias, params.alpha);
        continue;
      }
    }
    const net = model.convNetwork(index);
    if (!net) {
      throw new Error(`missing brain.js network for conv layer ${index}`);
    }
    current = brainConvForward(current, layer, net);
  }
  return current;
}
