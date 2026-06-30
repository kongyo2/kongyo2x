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
  const wBlocks = Math.floor(width / processSize) + (width % processSize === 0 ? 0 : 1);
  const hBlocks = Math.floor(height / processSize) + (height % processSize === 0 ? 0 : 1);
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
      const dstRow = dstPlane + (oy0 + y) * tw + ox0;
      for (let x = 0; x < bw; x++) {
        target.data[dstRow + x] = block.data[srcRow + x] as number;
      }
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

export function reconstructImageY(model: Kongyo2xModel, rgb: Tensor, options: ReconstructOptions = {}): Tensor {
  const { blockSize } = resolve(options);
  const offset = model.meta.offset;
  const p = paddingParams(rgb.width, rgb.height, offset, 1, blockSize);
  const padded = padEdge(rgb, p.padW1, p.padW2, p.padH1, p.padH2);
  const yuv = rgb2yuv(padded);
  const yPlane = getChannel(yuv, 0);
  const refined = reconstructNN(model, yPlane, 1, offset, blockSize);
  const yuvCrop = crop(yuv, p.padW1, p.padH1, p.padW1 + rgb.width, p.padH1 + rgb.height);
  const yCrop = clamp01(crop(refined, 0, 0, rgb.width, rgb.height));
  yuvCrop.data.set(yCrop.data, 0);
  return clamp01(yuv2rgb(yuvCrop));
}

export function reconstructImageRgb(model: Kongyo2xModel, rgb: Tensor, options: ReconstructOptions = {}): Tensor {
  const { blockSize } = resolve(options);
  const offset = model.meta.offset;
  const p = paddingParams(rgb.width, rgb.height, offset, 1, blockSize);
  const padded = padEdge(rgb, p.padW1, p.padW2, p.padH1, p.padH2);
  const refined = reconstructNN(model, padded, 1, offset, blockSize);
  return clamp01(crop(refined, 0, 0, rgb.width, rgb.height));
}

export function reconstructScaleY(
  model: Kongyo2xModel,
  scale: number,
  rgb: Tensor,
  options: ReconstructOptions = {},
): Tensor {
  const { blockSize } = resolve(options);
  const offset = model.meta.offset;
  const innerScale = model.meta.scaleFactor;
  const finalW = rgb.width * scale;
  const finalH = rgb.height * scale;
  const lanczos = rgb2yuv(resizeLanczos(rgb, finalW, finalH));

  let processed: Tensor;
  let nnInnerScale: number;
  if (model.meta.resize) {
    processed = rgb;
    nnInnerScale = innerScale;
  } else {
    processed = resizeNearest(rgb, finalW, finalH);
    nnInnerScale = 1;
  }
  const p = paddingParams(processed.width, processed.height, offset, nnInnerScale, blockSize);
  const padded = padEdge(processed, p.padW1, p.padW2, p.padH1, p.padH2);
  const yuv = rgb2yuv(padded);
  const yPlane = getChannel(yuv, 0);
  const refined = reconstructNN(model, yPlane, nnInnerScale, offset, blockSize);
  const yCrop = clamp01(crop(refined, 0, 0, finalW, finalH));
  lanczos.data.set(yCrop.data, 0);
  return clamp01(yuv2rgb(lanczos));
}

export function reconstructScaleRgb(
  model: Kongyo2xModel,
  scale: number,
  rgb: Tensor,
  options: ReconstructOptions = {},
): Tensor {
  const { blockSize } = resolve(options);
  const offset = model.meta.offset;
  const innerScale = model.meta.scaleFactor;
  let processed: Tensor;
  let nnInnerScale: number;
  if (model.meta.resize) {
    processed = rgb;
    nnInnerScale = innerScale;
  } else {
    processed = resizeNearest(rgb, rgb.width * scale, rgb.height * scale);
    nnInnerScale = 1;
  }
  const p = paddingParams(processed.width, processed.height, offset, nnInnerScale, blockSize);
  const padded = padEdge(processed, p.padW1, p.padW2, p.padH1, p.padH2);
  const refined = reconstructNN(model, padded, nnInnerScale, offset, blockSize);
  return clamp01(crop(refined, 0, 0, processed.width * nnInnerScale, processed.height * nnInnerScale));
}

export function reconstructImage(model: Kongyo2xModel, rgb: Tensor, options: ReconstructOptions = {}): Tensor {
  return model.isRgb ? reconstructImageRgb(model, rgb, options) : reconstructImageY(model, rgb, options);
}

export function reconstructScale(
  model: Kongyo2xModel,
  scale: number,
  rgb: Tensor,
  options: ReconstructOptions = {},
): Tensor {
  return model.isRgb ? reconstructScaleRgb(model, scale, rgb, options) : reconstructScaleY(model, scale, rgb, options);
}
