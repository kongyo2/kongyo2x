import type { RawLayerJSON, RawModelJSON } from "../src/model/types.js";
import { createTensor } from "../src/core/tensor.js";
import type { Tensor } from "../src/core/tensor.js";

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

function makeWeight4D(nOut: number, nIn: number, kH: number, kW: number, rand: () => number): number[][][][] {
  const w: number[][][][] = [];
  for (let o = 0; o < nOut; o++) {
    const wo: number[][][] = [];
    for (let i = 0; i < nIn; i++) {
      const wi: number[][] = [];
      for (let ky = 0; ky < kH; ky++) {
        const row: number[] = [];
        for (let kx = 0; kx < kW; kx++) {
          row.push((rand() - 0.5) * 0.4);
        }
        wi.push(row);
      }
      wo.push(wi);
    }
    w.push(wo);
  }
  return w;
}

export function convLayerJSON(nIn: number, nOut: number, kH: number, kW: number, rand: () => number): RawLayerJSON {
  return {
    class_name: "nn.SpatialConvolutionMM",
    nInputPlane: nIn,
    nOutputPlane: nOut,
    kW,
    kH,
    dW: 1,
    dH: 1,
    padW: 0,
    padH: 0,
    weight: makeWeight4D(nOut, nIn, kH, kW, rand),
    bias: Array.from({ length: nOut }, () => (rand() - 0.5) * 0.1),
  };
}

export function deconvLayerJSON(nIn: number, nOut: number, rand: () => number): RawLayerJSON {
  return {
    class_name: "nn.SpatialFullConvolution",
    nInputPlane: nIn,
    nOutputPlane: nOut,
    kW: 4,
    kH: 4,
    dW: 2,
    dH: 2,
    padW: 3,
    padH: 3,
    adjW: 0,
    adjH: 0,
    weight: makeWeight4D(nIn, nOut, 4, 4, rand),
    bias: Array.from({ length: nOut }, () => 0),
  };
}

export function vggLikeModel(channels: number, seed = 1): RawModelJSON {
  const rand = mulberry32(seed);
  const layers: RawModelJSON = [
    convLayerJSON(channels, 4, 3, 3, rand),
    convLayerJSON(4, 8, 3, 3, rand),
    convLayerJSON(8, channels, 3, 3, rand),
  ];
  (layers[0] as RawLayerJSON).model_config = {
    arch_name: "vgg_7",
    scale_factor: 1,
    channels,
    offset: 3,
  };
  return layers;
}

export function upconvLikeModel(channels: number, seed = 2): RawModelJSON {
  const rand = mulberry32(seed);
  const layers: RawModelJSON = [
    convLayerJSON(channels, 4, 3, 3, rand),
    convLayerJSON(4, 8, 3, 3, rand),
    deconvLayerJSON(8, channels, rand),
  ];
  (layers[0] as RawLayerJSON).model_config = {
    arch_name: "upconv_7",
    scale_factor: 2,
    channels,
    offset: 6,
    resize: true,
  };
  return layers;
}

export function randomImage(channels: number, height: number, width: number, seed = 7): Tensor {
  const rand = mulberry32(seed);
  const t = createTensor(channels, height, width);
  for (let i = 0; i < t.data.length; i++) {
    t.data[i] = rand();
  }
  return t;
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
