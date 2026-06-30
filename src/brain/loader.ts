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

type CreateContext = (width: number, height: number, options: Record<string, unknown>) => unknown;

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

function probeGpu(): boolean {
  if (process.env.KONGYO2X_DISABLE_GPU === "1" || process.env.KONGYO2X_GPU === "0") {
    return false;
  }
  try {
    const createContext = require("gl") as CreateContext;
    const context = createContext(1, 1, {}) as { getExtension(name: string): { destroy(): void } | null } | null;
    if (!context) {
      return false;
    }
    context.getExtension("STACKGL_destroy_context")?.destroy();
    return true;
  } catch {
    return false;
  }
}

let gpuAvailable: boolean | undefined;

export function isGpuAvailable(): boolean {
  if (gpuAvailable === undefined) {
    gpuAvailable = probeGpu();
  }
  return gpuAvailable;
}

let cached: BrainModule | undefined;

export function loadBrain(): BrainModule {
  if (cached) {
    return cached;
  }
  if (!isGpuAvailable()) {
    installGlStub();
  }
  cached = require("brain.js") as BrainModule;
  return cached;
}
