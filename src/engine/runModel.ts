import type { Tensor } from "../core/tensor.js";
import type { Waifu2xModel } from "../model/model.js";
import { spatialConvolutionMM } from "./conv.js";
import { spatialFullConvolution } from "./deconv.js";
import { brainConvForward } from "./brainConv.js";
import { applyLeakyReluInPlace } from "./leakyRelu.js";

export type Engine = "fast" | "brain";

const LEAKY_RELU_ALPHA = 0.1;

export function runModel(model: Waifu2xModel, input: Tensor, engine: Engine): Tensor {
  let current = input;
  for (let index = 0; index < model.layers.length; index++) {
    const layer = model.layers[index] as (typeof model.layers)[number];
    const isLast = model.isLastLayer(index);
    if (layer.kind === "deconv") {
      current = spatialFullConvolution(current, layer);
      continue;
    }
    if (engine === "brain") {
      const net = model.brainLayer(index);
      if (!net) {
        throw new Error(`missing brain network for layer ${index}`);
      }
      current = brainConvForward(current, layer, net);
    } else {
      current = spatialConvolutionMM(current, layer);
      if (!isLast) {
        applyLeakyReluInPlace(current, LEAKY_RELU_ALPHA);
      }
    }
  }
  return current;
}
