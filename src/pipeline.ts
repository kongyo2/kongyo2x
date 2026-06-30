import { readFile } from "node:fs/promises";
import { Waifu2xModel } from "./model/model.js";
import type { RawModelJSON } from "./model/types.js";
import type { Tensor } from "./core/tensor.js";
import { stackChannels } from "./core/tensor.js";
import { reconstructImage, reconstructScale } from "./reconstruct.js";
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

export function loadModelFromJson(raw: RawModelJSON): Waifu2xModel {
  return Waifu2xModel.fromJSON(raw);
}

export async function loadModelFile(path: string): Promise<Waifu2xModel> {
  const text = await readFile(path, "utf8");
  return Waifu2xModel.fromJSON(JSON.parse(text) as RawModelJSON);
}

export function denoise(model: Waifu2xModel, input: ImageInput, options: ReconstructOptions = {}): ProcessedImage {
  const rgb = reconstructImage(model, input.rgb, options);
  return input.alpha ? { rgb, alpha: input.alpha } : { rgb };
}

function upscaleAlpha(
  scaleModel: Waifu2xModel,
  scale: number,
  alpha: Tensor,
  mode: AlphaScaleMode,
  options: ReconstructOptions,
): Tensor {
  const finalW = alpha.width * scale;
  const finalH = alpha.height * scale;
  if (mode === "lanczos") {
    return resizeLanczos(alpha, finalW, finalH);
  }
  const rgb = stackChannels([alpha, alpha, alpha]);
  const scaled = reconstructScale(scaleModel, scale, rgb, options);
  return rgb2y(scaled);
}

export function scaleImage(
  model: Waifu2xModel,
  scale: number,
  input: ImageInput,
  options: ScaleOptions = {},
  alphaModel: Waifu2xModel = model,
): ProcessedImage {
  const reconstructOptions: ReconstructOptions = {};
  if (options.blockSize !== undefined) {
    reconstructOptions.blockSize = options.blockSize;
  }
  if (options.engine !== undefined) {
    reconstructOptions.engine = options.engine;
  }
  const bordered = input.alpha ? makeBorder(input.rgb, input.alpha, alphaModel.meta.offset) : input.rgb;
  const rgb = reconstructScale(model, scale, bordered, reconstructOptions);
  if (!input.alpha) {
    return { rgb };
  }
  const alpha = upscaleAlpha(alphaModel, scale, input.alpha, options.alphaScale ?? "model", reconstructOptions);
  return { rgb, alpha };
}

export function denoiseThenScale(
  noiseModel: Waifu2xModel,
  scaleModel: Waifu2xModel,
  scale: number,
  input: ImageInput,
  options: ScaleOptions = {},
): ProcessedImage {
  const reconstructOptions: ReconstructOptions = {};
  if (options.blockSize !== undefined) {
    reconstructOptions.blockSize = options.blockSize;
  }
  if (options.engine !== undefined) {
    reconstructOptions.engine = options.engine;
  }
  const bordered = input.alpha ? makeBorder(input.rgb, input.alpha, scaleModel.meta.offset) : input.rgb;
  const denoised = reconstructImage(noiseModel, bordered, reconstructOptions);
  const rgb = reconstructScale(scaleModel, scale, denoised, reconstructOptions);
  if (!input.alpha) {
    return { rgb };
  }
  const alpha = upscaleAlpha(scaleModel, scale, input.alpha, options.alphaScale ?? "model", reconstructOptions);
  return { rgb, alpha };
}

export function noiseScaleCombined(
  noiseScaleModel: Waifu2xModel,
  scale: number,
  input: ImageInput,
  options: ScaleOptions = {},
  scaleModel: Waifu2xModel = noiseScaleModel,
): ProcessedImage {
  return scaleImage(noiseScaleModel, scale, input, options, scaleModel);
}
