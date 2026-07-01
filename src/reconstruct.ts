import { clamp01, createTensor, getChannel } from "./core/tensor.js";
import type { Tensor } from "./core/tensor.js";
import type { Kongyo2xModel } from "./model/model.js";
import { runModel } from "./engine/runModel.js";
import { rgb2yuv, yuv2rgb } from "./image/color.js";
import { crop, padEdge } from "./image/pad.js";
import { resizeLanczos, resizeNearest } from "./image/resize.js";

export interface ReconstructOptions {
  blockSize?: number;
}

interface Padding {
  innerScale: number;
  inputOffset: number;
  padW1: number;
  padW2: number;
  padH1: number;
  padH2: number;
}

function paddingParams(width: number, height: number, offset: number, innerScale: number, blockSize: number): Padding {
  const inputOffset = Math.ceil(offset / innerScale);
  const processSize = blockSize - inputOffset * 2;
  if (processSize <= 0) {
    throw new Error(`block size ${blockSize} is too small for model offset ${offset}`);
  }
  const wBlocks = Math.ceil(width / processSize);
  const hBlocks = Math.ceil(height / processSize);
  const paddedW = wBlocks * processSize + inputOffset * 2;
  const paddedH = hBlocks * processSize + inputOffset * 2;
  return {
    innerScale,
    inputOffset,
    padW1: inputOffset,
    padW2: paddedW - inputOffset - width,
    padH1: inputOffset,
    padH2: paddedH - inputOffset - height,
  };
}

function placeBlock(target: Tensor, block: Tensor, oy0: number, ox0: number): void {
  const { channels, height: bh, width: bw } = block;
  const tw = target.width;
  const th = target.height;
  for (let c = 0; c < channels; c++) {
    const srcPlane = c * bh * bw;
    const dstPlane = c * th * tw;
    for (let y = 0; y < bh; y++) {
      const srcRow = srcPlane + y * bw;
      target.data.set(block.data.subarray(srcRow, srcRow + bw), dstPlane + (oy0 + y) * tw + ox0);
    }
  }
}

function reconstructNN(model: Kongyo2xModel, x: Tensor, innerScale: number, offset: number, blockSize: number): Tensor {
  const newX = createTensor(x.channels, x.height * innerScale, x.width * innerScale);
  const inputBlockSize = blockSize;
  const outputSize = blockSize * innerScale - offset * 2;
  const inputOffset = Math.ceil(offset / innerScale);
  const step = inputBlockSize - inputOffset * 2;
  for (let i0 = 0; i0 + inputBlockSize <= x.height; i0 += step) {
    for (let j0 = 0; j0 + inputBlockSize <= x.width; j0 += step) {
      const block = crop(x, j0, i0, j0 + inputBlockSize, i0 + inputBlockSize);
      const out = runModel(model, block);
      if (out.height !== outputSize || out.width !== outputSize) {
        throw new Error(`model produced ${out.width}x${out.height}, expected ${outputSize}x${outputSize}`);
      }
      placeBlock(newX, out, i0 * innerScale, j0 * innerScale);
    }
  }
  return newX;
}

function resolve(options: ReconstructOptions): { blockSize: number } {
  return { blockSize: options.blockSize ?? 128 };
}

function scaledSize(size: number, scale: number): number {
  return Math.max(1, Math.round(size * scale));
}

function validateScale(scale: number): void {
  if (!Number.isFinite(scale) || scale <= 0) {
    throw new Error(`invalid scale factor ${scale} (expected a finite value > 0)`);
  }
}

function checkResizeModel(model: Kongyo2xModel): number {
  const innerScale = model.meta.scaleFactor;
  if (!Number.isInteger(innerScale) || innerScale < 2) {
    throw new Error(`resize model must declare an integer scaleFactor >= 2, received ${innerScale}`);
  }
  if (model.meta.offset % innerScale !== 0) {
    // Blocks are placed on an innerScale-aligned grid, so an unaligned offset
    // would shift every block by (ceil(offset/innerScale) * innerScale - offset)
    // output pixels and produce visible seams.
    throw new Error(`resize model offset ${model.meta.offset} must be a multiple of scaleFactor ${innerScale}`);
  }
  return innerScale;
}

/** One model pass of a resize model over an RGB tensor: multiplies both dimensions by innerScale. */
function refinePassRgb(model: Kongyo2xModel, rgb: Tensor, innerScale: number, blockSize: number): Tensor {
  const offset = model.meta.offset;
  const p = paddingParams(rgb.width, rgb.height, offset, innerScale, blockSize);
  const padded = padEdge(rgb, p.padW1, p.padW2, p.padH1, p.padH2);
  const refined = reconstructNN(model, padded, innerScale, offset, blockSize);
  return clamp01(crop(refined, 0, 0, rgb.width * innerScale, rgb.height * innerScale));
}

/**
 * One model pass of a resize luma model: the Y channel comes from the network,
 * chroma from a Lanczos resample of the input.
 */
function refinePassY(model: Kongyo2xModel, rgb: Tensor, innerScale: number, blockSize: number): Tensor {
  const offset = model.meta.offset;
  const outW = rgb.width * innerScale;
  const outH = rgb.height * innerScale;
  const lanczos = rgb2yuv(resizeLanczos(rgb, outW, outH));
  const p = paddingParams(rgb.width, rgb.height, offset, innerScale, blockSize);
  const padded = padEdge(rgb, p.padW1, p.padW2, p.padH1, p.padH2);
  const yPlane = getChannel(rgb2yuv(padded), 0);
  const refined = reconstructNN(model, yPlane, innerScale, offset, blockSize);
  const yCrop = clamp01(crop(refined, 0, 0, outW, outH));
  lanczos.data.set(yCrop.data, 0);
  return clamp01(yuv2rgb(lanczos));
}

/** Pre-upscale (nearest) to the target size, then refine with a non-resize model at 1x. */
function refineAtSizeRgb(model: Kongyo2xModel, rgb: Tensor, outW: number, outH: number, blockSize: number): Tensor {
  const offset = model.meta.offset;
  const processed = resizeNearest(rgb, outW, outH);
  const p = paddingParams(outW, outH, offset, 1, blockSize);
  const padded = padEdge(processed, p.padW1, p.padW2, p.padH1, p.padH2);
  const refined = reconstructNN(model, padded, 1, offset, blockSize);
  return clamp01(crop(refined, 0, 0, outW, outH));
}

function refineAtSizeY(model: Kongyo2xModel, rgb: Tensor, outW: number, outH: number, blockSize: number): Tensor {
  const offset = model.meta.offset;
  const lanczos = rgb2yuv(resizeLanczos(rgb, outW, outH));
  const processed = resizeNearest(rgb, outW, outH);
  const p = paddingParams(outW, outH, offset, 1, blockSize);
  const padded = padEdge(processed, p.padW1, p.padW2, p.padH1, p.padH2);
  const yPlane = getChannel(rgb2yuv(padded), 0);
  const refined = reconstructNN(model, yPlane, 1, offset, blockSize);
  const yCrop = clamp01(crop(refined, 0, 0, outW, outH));
  lanczos.data.set(yCrop.data, 0);
  return clamp01(yuv2rgb(lanczos));
}

type RefinePass = (model: Kongyo2xModel, rgb: Tensor, innerScale: number, blockSize: number) => Tensor;
type RefineAtSize = (model: Kongyo2xModel, rgb: Tensor, outW: number, outH: number, blockSize: number) => Tensor;

/**
 * Scale `rgb` by an arbitrary factor. Non-resize models refine a nearest
 * pre-upscale of the exact target size. Resize models run repeatedly (each
 * pass multiplies the size by `meta.scaleFactor`) until the target is reached,
 * then the result is Lanczos-resampled to the exact target when they differ —
 * so factors like 3, 4, or 1.5 work with a 2x model.
 */
function reconstructWith(
  pass: RefinePass,
  atSize: RefineAtSize,
  model: Kongyo2xModel,
  scale: number,
  rgb: Tensor,
  options: ReconstructOptions,
): Tensor {
  validateScale(scale);
  const { blockSize } = resolve(options);
  const finalW = scaledSize(rgb.width, scale);
  const finalH = scaledSize(rgb.height, scale);
  if (!model.meta.resize) {
    return atSize(model, rgb, finalW, finalH, blockSize);
  }
  const innerScale = checkResizeModel(model);
  let current = rgb;
  let achieved = 1;
  do {
    current = pass(model, current, innerScale, blockSize);
    achieved *= innerScale;
  } while (achieved < scale);
  if (current.width !== finalW || current.height !== finalH) {
    current = clamp01(resizeLanczos(current, finalW, finalH));
  }
  return current;
}

export function reconstructScaleY(
  model: Kongyo2xModel,
  scale: number,
  rgb: Tensor,
  options: ReconstructOptions = {},
): Tensor {
  return reconstructWith(refinePassY, refineAtSizeY, model, scale, rgb, options);
}

export function reconstructScaleRgb(
  model: Kongyo2xModel,
  scale: number,
  rgb: Tensor,
  options: ReconstructOptions = {},
): Tensor {
  return reconstructWith(refinePassRgb, refineAtSizeRgb, model, scale, rgb, options);
}

export function reconstructScale(
  model: Kongyo2xModel,
  scale: number,
  rgb: Tensor,
  options: ReconstructOptions = {},
): Tensor {
  return model.isRgb ? reconstructScaleRgb(model, scale, rgb, options) : reconstructScaleY(model, scale, rgb, options);
}
