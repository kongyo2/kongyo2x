import { createRequire } from "node:module";
import { convNetworkJSON, LEAKY_RELU_ALPHA, IDENTITY_ALPHA } from "../src/index.js";
import type { ConvLayerJSON, DeconvLayerJSON, Kongyo2xModelJSON, ModelLayerJSON } from "../src/index.js";
import { createTensor } from "../src/core/tensor.js";
import type { Tensor } from "../src/core/tensor.js";

const require = createRequire(import.meta.url);

/**
 * brain.js is an optionalDependency: its native gl build fails on headless
 * machines and npm then skips the package entirely. Tests that compare against
 * real brain.js behavior are skipped in that case instead of failing.
 */
export function brainAvailable(): boolean {
  try {
    require.resolve("brain.js");
    return true;
  } catch {
    return false;
  }
}

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randomWeights(nOut: number, inSize: number, rand: () => number): number[][] {
  const weights: number[][] = [];
  for (let o = 0; o < nOut; o++) {
    const row = new Array<number>(inSize);
    for (let k = 0; k < inSize; k++) {
      row[k] = (rand() - 0.5) * 0.4;
    }
    weights.push(row);
  }
  return weights;
}

export function convLayerJSON(
  nIn: number,
  nOut: number,
  kH: number,
  kW: number,
  rand: () => number,
  leakyReluAlpha: number,
): ConvLayerJSON {
  const weights = randomWeights(nOut, nIn * kH * kW, rand);
  const bias = Array.from({ length: nOut }, () => (rand() - 0.5) * 0.1);
  return {
    kind: "conv",
    kernelWidth: kW,
    kernelHeight: kH,
    strideX: 1,
    strideY: 1,
    padX: 0,
    padY: 0,
    network: convNetworkJSON(weights, bias, leakyReluAlpha),
  };
}

export function deconvLayerJSON(nIn: number, nOut: number, rand: () => number): DeconvLayerJSON {
  const kH = 4;
  const kW = 4;
  const weights = new Array<number>(nIn * nOut * kH * kW);
  for (let i = 0; i < weights.length; i++) {
    weights[i] = (rand() - 0.5) * 0.4;
  }
  return {
    kind: "deconv",
    inputPlanes: nIn,
    outputPlanes: nOut,
    kernelWidth: kW,
    kernelHeight: kH,
    strideX: 2,
    strideY: 2,
    padX: 3,
    padY: 3,
    adjX: 0,
    adjY: 0,
    weights,
    bias: Array.from({ length: nOut }, () => 0),
  };
}

export function vggLikeModel(channels: number, seed = 1): Kongyo2xModelJSON {
  const rand = mulberry32(seed);
  const layers: ModelLayerJSON[] = [
    convLayerJSON(channels, 4, 3, 3, rand, LEAKY_RELU_ALPHA),
    convLayerJSON(4, 8, 3, 3, rand, LEAKY_RELU_ALPHA),
    convLayerJSON(8, channels, 3, 3, rand, IDENTITY_ALPHA),
  ];
  return {
    type: "kongyo2x",
    version: 1,
    meta: { archName: "vgg_7", channels, offset: 3, scaleFactor: 1, resize: false },
    layers,
  };
}

export function upconvLikeModel(channels: number, seed = 2): Kongyo2xModelJSON {
  const rand = mulberry32(seed);
  const layers: ModelLayerJSON[] = [
    convLayerJSON(channels, 4, 3, 3, rand, LEAKY_RELU_ALPHA),
    convLayerJSON(4, 8, 3, 3, rand, LEAKY_RELU_ALPHA),
    deconvLayerJSON(8, channels, rand),
  ];
  return {
    type: "kongyo2x",
    version: 1,
    meta: { archName: "upconv_7", channels, offset: 6, scaleFactor: 2, resize: true },
    layers,
  };
}

export function randomImage(channels: number, height: number, width: number, seed = 7): Tensor {
  const rand = mulberry32(seed);
  const t = createTensor(channels, height, width);
  for (let i = 0; i < t.data.length; i++) {
    t.data[i] = rand();
  }
  return t;
}

export function hasNaN(t: Tensor): boolean {
  for (let i = 0; i < t.data.length; i++) {
    if (Number.isNaN(t.data[i] as number)) {
      return true;
    }
  }
  return false;
}

export function maxAbsDiff(a: Tensor, b: Tensor): number {
  if (a.data.length !== b.data.length) {
    throw new Error("tensor size mismatch");
  }
  let max = 0;
  for (let i = 0; i < a.data.length; i++) {
    const d = Math.abs((a.data[i] as number) - (b.data[i] as number));
    if (d > max) {
      max = d;
    }
  }
  return max;
}
