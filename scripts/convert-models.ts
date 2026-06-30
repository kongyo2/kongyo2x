import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { convNetworkJSON, LEAKY_RELU_ALPHA, IDENTITY_ALPHA } from "../src/index.js";
import type { ConvLayerJSON, DeconvLayerJSON, Kongyo2xModelJSON, ModelLayerJSON, ModelMeta } from "../src/index.js";

interface UpstreamLayer {
  class_name?: string;
  nInputPlane: number;
  nOutputPlane: number;
  kW: number;
  kH: number;
  dW?: number;
  dH?: number;
  padW?: number;
  padH?: number;
  adjW?: number;
  adjH?: number;
  weight: number[][][][];
  bias?: number[];
  model_config?: {
    arch_name?: string;
    scale_factor?: number;
    channels?: number;
    offset?: number;
    resize?: boolean;
  };
}

type UpstreamModel = UpstreamLayer[];

const DECONV = "nn.SpatialFullConvolution";

function biasArray(layer: UpstreamLayer): number[] {
  const bias = new Array<number>(layer.nOutputPlane);
  for (let o = 0; o < layer.nOutputPlane; o++) {
    bias[o] = layer.bias?.[o] ?? 0;
  }
  return bias;
}

function convLayerJSON(layer: UpstreamLayer, leakyReluAlpha: number): ConvLayerJSON {
  const { nInputPlane, nOutputPlane, kH, kW } = layer;
  const weights: number[][] = [];
  for (let o = 0; o < nOutputPlane; o++) {
    const row = new Array<number>(nInputPlane * kH * kW);
    let k = 0;
    for (let i = 0; i < nInputPlane; i++) {
      for (let ky = 0; ky < kH; ky++) {
        for (let kx = 0; kx < kW; kx++) {
          row[k++] = layer.weight[o][i][ky][kx];
        }
      }
    }
    weights.push(row);
  }
  return {
    kind: "conv",
    kernelWidth: kW,
    kernelHeight: kH,
    strideX: layer.dW ?? 1,
    strideY: layer.dH ?? 1,
    padX: layer.padW ?? 0,
    padY: layer.padH ?? 0,
    network: convNetworkJSON(weights, biasArray(layer), leakyReluAlpha),
  };
}

function deconvLayerJSON(layer: UpstreamLayer): DeconvLayerJSON {
  const { nInputPlane, nOutputPlane, kH, kW } = layer;
  const weights = new Array<number>(nInputPlane * nOutputPlane * kH * kW);
  let k = 0;
  for (let i = 0; i < nInputPlane; i++) {
    for (let o = 0; o < nOutputPlane; o++) {
      for (let ky = 0; ky < kH; ky++) {
        for (let kx = 0; kx < kW; kx++) {
          weights[k++] = layer.weight[i][o][ky][kx];
        }
      }
    }
  }
  return {
    kind: "deconv",
    inputPlanes: nInputPlane,
    outputPlanes: nOutputPlane,
    kernelWidth: kW,
    kernelHeight: kH,
    strideX: layer.dW ?? 1,
    strideY: layer.dH ?? 1,
    padX: layer.padW ?? 0,
    padY: layer.padH ?? 0,
    adjX: layer.adjW ?? 0,
    adjY: layer.adjH ?? 0,
    weights,
    bias: biasArray(layer),
  };
}

function deriveMeta(raw: UpstreamModel): ModelMeta {
  const config = raw[0]?.model_config;
  const last = raw[raw.length - 1];
  const hasDeconv = raw.some((l) => l.class_name === DECONV);
  const fallbackChannels = last ? last.nOutputPlane : 1;
  const fallbackOffset = raw
    .filter((l) => l.class_name !== DECONV)
    .reduce((acc, l) => acc + Math.floor((l.kW - 1) / 2), 0);
  const scaleFactor = config?.scale_factor ?? (hasDeconv ? 2 : 1);
  return {
    archName: config?.arch_name ?? (hasDeconv ? "upconv_7" : "vgg_7"),
    channels: config?.channels ?? fallbackChannels,
    offset: config?.offset ?? fallbackOffset,
    scaleFactor,
    resize: config?.resize ?? scaleFactor > 1,
  };
}

export function convertModel(raw: UpstreamModel): Kongyo2xModelJSON {
  if (!Array.isArray(raw) || raw.length === 0) {
    throw new Error("upstream model must be a non-empty array of layers");
  }
  const layers: ModelLayerJSON[] = raw.map((layer, index) => {
    if (layer.class_name === DECONV) {
      return deconvLayerJSON(layer);
    }
    const isLast = index === raw.length - 1;
    return convLayerJSON(layer, isLast ? IDENTITY_ALPHA : LEAKY_RELU_ALPHA);
  });
  return { type: "kongyo2x", version: 1, meta: deriveMeta(raw), layers };
}

export async function convertFile(srcPath: string, destPath: string): Promise<void> {
  const raw = JSON.parse(await readFile(srcPath, "utf8")) as UpstreamModel;
  const model = convertModel(raw);
  await mkdir(dirname(destPath), { recursive: true });
  await writeFile(destPath, JSON.stringify(model));
}

const BASE = "https://raw.githubusercontent.com/nagadomi/waifu2x/master/models";

const CANDIDATE_FILES = [
  "noise0_model.json",
  "noise1_model.json",
  "noise2_model.json",
  "noise3_model.json",
  "scale2.0x_model.json",
  "noise0_scale2.0x_model.json",
  "noise1_scale2.0x_model.json",
  "noise2_scale2.0x_model.json",
  "noise3_scale2.0x_model.json",
];

const DEFAULT_DIRS = ["vgg_7/art_y", "vgg_7/art", "upconv_7/art"];

async function convertDir(dir: string): Promise<void> {
  for (const file of CANDIDATE_FILES) {
    const url = `${BASE}/${dir}/${file}`;
    const response = await fetch(url, { headers: { "User-Agent": "kongyo2x/0.1" } });
    if (!response.ok) {
      continue;
    }
    const raw = (await response.json()) as UpstreamModel;
    const model = convertModel(raw);
    const target = join("models", dir, file);
    await mkdir(dirname(target), { recursive: true });
    const json = JSON.stringify(model);
    await writeFile(target, json);
    process.stdout.write(`converted ${target} (${(json.length / 1024).toFixed(0)} KB)\n`);
  }
}

async function main(): Promise<void> {
  const dirs = process.argv.slice(2);
  const targets = dirs.length > 0 ? dirs : DEFAULT_DIRS;
  for (const dir of targets) {
    process.stdout.write(`converting model set: ${dir}\n`);
    await convertDir(dir);
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error: unknown) => {
    process.stderr.write(`error: ${error instanceof Error ? error.message : String(error)}\n`);
    process.exit(1);
  });
}
