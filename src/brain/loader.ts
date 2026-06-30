import Module from "node:module";
import { createRequire } from "node:module";

export interface BrainNetworkJSON {
  type: "NeuralNetwork";
  sizes: number[];
  layers: Array<{ weights: number[][]; biases: number[] }>;
  inputLookup: null;
  inputLookupLength: number;
  outputLookup: null;
  outputLookupLength: number;
  options: {
    inputSize: number;
    outputSize: number;
    binaryThresh: number;
    activation: string;
    leakyReluAlpha: number;
    hiddenLayers: number[];
  };
  trainOpts: { activation: string; leakyReluAlpha: number; timeout: string };
}

export interface BrainNeuralNetwork {
  fromJSON(json: BrainNetworkJSON): BrainNeuralNetwork;
  toJSON(): BrainNetworkJSON;
  run(input: Float32Array): Float32Array;
  trainOpts: { leakyReluAlpha: number };
}

interface BrainModule {
  NeuralNetwork: new () => BrainNeuralNetwork;
}

interface NodeModuleInternals {
  _load(request: string, parent: unknown, isMain: boolean): unknown;
}

const require = createRequire(import.meta.url);

function installGlStub(): void {
  const internals = Module as unknown as NodeModuleInternals;
  const original = internals._load.bind(internals);
  internals._load = (request: string, parent: unknown, isMain: boolean): unknown => {
    if (request === "gl") {
      return function glUnavailable(): never {
        throw new Error("headless-gl is not available; kongyo2x runs brain.js on the CPU only.");
      };
    }
    return original(request, parent, isMain);
  };
}

let cached: BrainModule | undefined;

function brainInstalled(): boolean {
  try {
    require.resolve("brain.js");
    return true;
  } catch {
    return false;
  }
}

export function loadBrain(): BrainModule {
  if (cached) {
    return cached;
  }
  try {
    cached = require("brain.js") as BrainModule;
  } catch (error) {
    // brain.js loads gl eagerly; the stub lets it run on the CPU when the gl
    // native build is missing. When brain.js itself is absent the retry can't
    // succeed, so skip the stub rather than shadow `gl` process-wide for code
    // that has nothing to do with kongyo2x.
    if (!brainInstalled()) {
      throw error;
    }
    installGlStub();
    cached = require("brain.js") as BrainModule;
  }
  return cached;
}
