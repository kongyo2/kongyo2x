import type { Tensor } from "../core/tensor.js";
import type { Kongyo2xModel } from "../model/model.js";
import type { ModelLayer } from "../model/types.js";
import { spatialFullConvolution } from "./deconv.js";
import { brainConvForward } from "./brainConv.js";
import { cpuConvForward } from "./cpuConv.js";
import { gpuConvForward, isGpuAvailable, disableGpu } from "./gpuConv.js";
import { isWasmAvailable } from "../wasm/loader.js";

export function runModel(model: Kongyo2xModel, input: Tensor): Tensor {
  let useGpu = isGpuAvailable();
  const useWasm = isWasmAvailable();
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
    // The Wasm CPU convolution outruns brain.js's per-patch forward, so prefer
    // it whenever the layer exposes raw parameters and Wasm is available.
    if (useWasm) {
      const params = model.convParams(index);
      if (params) {
        current = cpuConvForward(current, layer, params.weights, params.bias, params.alpha);
        continue;
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
