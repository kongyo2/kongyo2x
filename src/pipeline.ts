import { readFile } from "node:fs/promises";
import { Kongyo2xModel } from "./model/model.js";
import type { Kongyo2xModelJSON } from "./model/types.js";
import type { Tensor } from "./core/tensor.js";
import { clamp01, stackChannels } from "./core/tensor.js";
import { reconstructScale } from "./reconstruct.js";
import type { ReconstructOptions } from "./reconstruct.js";
import { rgb2y } from "./image/color.js";
import { makeBorder } from "./image/alpha.js";
import { resizeLanczos } from "./image/resize.js";

export interface ProcessedImage {
  rgb: Tensor;
  alpha?: Tensor;
}

export interface ImageInput {
  rgb: Tensor;
  alpha?: Tensor;
}

export type AlphaScaleMode = "model" | "lanczos";

export interface ScaleOptions extends ReconstructOptions {
  alphaScale?: AlphaScaleMode;
}

export function loadModelFromJson(json: Kongyo2xModelJSON): Kongyo2xModel {
  return Kongyo2xModel.fromJSON(json);
}

export async function loadModelFile(path: string): Promise<Kongyo2xModel> {
  const text = await readFile(path, "utf8");
  return Kongyo2xModel.fromJSON(JSON.parse(text) as Kongyo2xModelJSON);
}

function upscaleAlpha(
  scaleModel: Kongyo2xModel,
  scale: number,
  alpha: Tensor,
  mode: AlphaScaleMode,
  options: ReconstructOptions,
): Tensor {
  // Round exactly like reconstructScale so the alpha plane always matches the
  // RGB output dimensions, fractional scale factors included.
  const finalW = Math.max(1, Math.round(alpha.width * scale));
  const finalH = Math.max(1, Math.round(alpha.height * scale));
  if (mode === "lanczos") {
    return clamp01(resizeLanczos(alpha, finalW, finalH));
  }
  const rgb = stackChannels([alpha, alpha, alpha]);
  const scaled = reconstructScale(scaleModel, scale, rgb, options);
  return rgb2y(scaled);
}

function reconstructOptionsOf(options: ScaleOptions): ReconstructOptions {
  return options.blockSize !== undefined ? { blockSize: options.blockSize } : {};
}

export function scaleImage(
  model: Kongyo2xModel,
  scale: number,
  input: ImageInput,
  options: ScaleOptions = {},
  alphaModel: Kongyo2xModel = model,
): ProcessedImage {
  const reconstructOptions = reconstructOptionsOf(options);
  const bordered = input.alpha ? makeBorder(input.rgb, input.alpha, alphaModel.meta.offset) : input.rgb;
  const rgb = reconstructScale(model, scale, bordered, reconstructOptions);
  if (!input.alpha) {
    return { rgb };
  }
  const alpha = upscaleAlpha(alphaModel, scale, input.alpha, options.alphaScale ?? "model", reconstructOptions);
  return { rgb, alpha };
}
