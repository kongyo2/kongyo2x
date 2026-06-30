import { loadBrain } from "./loader.js";
import type { BrainNetworkJSON, BrainNeuralNetwork } from "./loader.js";

export type { BrainNetworkJSON, BrainNeuralNetwork } from "./loader.js";

export const LEAKY_RELU_ALPHA = 0.1;
/** leaky-relu with alpha 1 leaves every value untouched, i.e. a linear (identity) activation. */
export const IDENTITY_ALPHA = 1.0;

export function convNetworkJSON(weights: number[][], bias: number[], leakyReluAlpha: number): BrainNetworkJSON {
  const outputSize = weights.length;
  const inputSize = outputSize > 0 ? (weights[0]?.length ?? 0) : 0;
  return {
    type: "NeuralNetwork",
    sizes: [inputSize, outputSize],
    layers: [
      { weights: [], biases: [] },
      { weights, biases: bias },
    ],
    inputLookup: null,
    inputLookupLength: 0,
    outputLookup: null,
    outputLookupLength: 0,
    options: {
      inputSize,
      outputSize,
      binaryThresh: 0.5,
      activation: "leaky-relu",
      leakyReluAlpha,
      hiddenLayers: [],
    },
    trainOpts: { activation: "leaky-relu", leakyReluAlpha, timeout: "Infinity" },
  };
}

export function buildNetwork(json: BrainNetworkJSON): BrainNeuralNetwork {
  const { NeuralNetwork } = loadBrain();
  const net = new NeuralNetwork();
  const alpha = json.trainOpts.leakyReluAlpha;
  net.fromJSON({ ...json, trainOpts: { ...json.trainOpts, leakyReluAlpha: LEAKY_RELU_ALPHA } });
  net.trainOpts.leakyReluAlpha = alpha;
  return net;
}
