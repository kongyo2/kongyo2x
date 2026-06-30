import type { Tensor } from "../core/tensor.js";
import type { Kongyo2xModel } from "../model/model.js";
import type { ModelLayer } from "../model/types.js";
import { spatialFullConvolution } from "./deconv.js";
import { brainConvForward } from "./brainConv.js";

export function runModel(model: Kongyo2xModel, input: Tensor): Tensor {
  let current = input;
  for (let index = 0; index < model.layers.length; index++) {
    const layer = model.layers[index] as ModelLayer;
    if (layer.kind === "deconv") {
      current = spatialFullConvolution(current, layer);
      continue;
    }
    const net = model.convNetwork(index);
    if (!net) {
      throw new Error(`missing brain.js network for conv layer ${index}`);
    }
    current = brainConvForward(current, layer, net);
  }
  return current;
}
