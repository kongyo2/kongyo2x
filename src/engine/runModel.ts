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
    const params = model.convParams(index);
    if (params) {
      if (useGpu) {
        try {
          current = gpuConvForward(current, layer, params.weights, params.bias, params.alpha);
          continue;
        } catch {
          disableGpu();
          useGpu = false;
        }
      }
      // Direct convolution over the raw parameters: Wasm when available, with
      // a pure-TS fallback inside. Both outrun brain.js's per-patch forward.
      current = cpuConvForward(current, layer, params.weights, params.bias, params.alpha);
      continue;
    }
    // Layers whose serialized network the fast engines cannot express (e.g. a
    // different activation) still run through brain.js when it is installed.
    const net = model.convNetwork(index);
    if (net) {
      current = brainConvForward(current, layer, net);
      continue;
    }
    throw new Error(
      `conv layer ${index} has no runnable backend: neither raw parameters nor a brain.js network are available`,
    );
  }
  return current;
}
