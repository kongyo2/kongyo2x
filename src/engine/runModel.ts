import type { Tensor } from "../core/tensor.js";
import type { Kongyo2xModel } from "../model/model.js";
import type { ModelLayer } from "../model/types.js";
import { spatialFullConvolution } from "./deconv.js";
import { brainConvForward } from "./brainConv.js";
import { cpuConvForward } from "./cpuConv.js";
import { gpuConvForward, isGpuAvailable, disableGpu } from "./gpuConv.js";

export function runModel(model: Kongyo2xModel, input: Tensor): Tensor {
  let useGpu = isGpuAvailable();
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
        try {
          current = gpuConvForward(current, layer, params.weights, params.bias, params.alpha);
          continue;
        } catch {
          disableGpu();
          useGpu = false;
        }
      }
    }
    const net = model.convNetwork(index);
    if (net) {
      current = brainConvForward(current, layer, net);
      continue;
    }
    const params = model.convParams(index);
    if (params) {
      current = cpuConvForward(current, layer, params.weights, params.bias, params.alpha);
      continue;
    }
    throw new Error(`conv layer ${index} has no runnable backend: neither brain.js nor CPU parameters are available`);
  }
  return current;
}
