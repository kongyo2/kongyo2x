import { readFile, writeFile } from "node:fs/promises";
import { PNG } from "pngjs";
import jpeg from "jpeg-js";
import { createTensor } from "../core/tensor.js";
import type { Tensor } from "../core/tensor.js";

const CLIP_EPS8 = (1.0 / 255.0) * 0.5 - 1e-7 * (1.0 / 255.0) * 0.5;

export interface LoadedImage {
  rgb: Tensor;
  alpha?: Tensor;
}

function rgbaToTensors(width: number, height: number, data: Uint8Array | Buffer): LoadedImage {
  const size = width * height;
  const rgb = createTensor(3, height, width);
  const rgbData = rgb.data;
  const alphaData = new Float32Array(size);
  let hasAlpha = false;
  for (let p = 0; p < size; p++) {
    const o = p * 4;
    rgbData[p] = (data[o] as number) / 255;
    rgbData[size + p] = (data[o + 1] as number) / 255;
    rgbData[2 * size + p] = (data[o + 2] as number) / 255;
    const a = data[o + 3] as number;
    alphaData[p] = a / 255;
    if (a !== 255) {
      hasAlpha = true;
    }
  }
  if (!hasAlpha) {
    return { rgb };
  }
  return { rgb, alpha: createTensorFrom(alphaData, height, width) };
}

function createTensorFrom(data: Float32Array, height: number, width: number): Tensor {
  return { channels: 1, height, width, data };
}

export function decodeImage(buffer: Buffer): LoadedImage {
  if (buffer.length >= 2 && buffer[0] === 0x89 && buffer[1] === 0x50) {
    const png = PNG.sync.read(buffer);
    return rgbaToTensors(png.width, png.height, png.data);
  }
  if (buffer.length >= 2 && buffer[0] === 0xff && buffer[1] === 0xd8) {
    const decoded = jpeg.decode(buffer, { useTArray: true, formatAsRGBA: true });
    return rgbaToTensors(decoded.width, decoded.height, decoded.data);
  }
  throw new Error("unsupported image format (expected PNG or JPEG)");
}

export async function loadImage(path: string): Promise<LoadedImage> {
  const buffer = await readFile(path);
  return decodeImage(buffer);
}

function floatToByte(v: number): number {
  const scaled = (v + CLIP_EPS8) * 255;
  if (scaled <= 0) {
    return 0;
  }
  if (scaled >= 255) {
    return 255;
  }
  return Math.floor(scaled);
}

export function encodePng(rgb: Tensor, alpha?: Tensor): Buffer {
  if (rgb.channels !== 3) {
    throw new Error("encodePng expects a 3-channel RGB tensor");
  }
  const { width, height } = rgb;
  const size = width * height;
  if (alpha && (alpha.width !== width || alpha.height !== height)) {
    throw new Error("alpha dimensions must match the RGB image");
  }
  const png = new PNG({ width, height });
  const out = png.data;
  const src = rgb.data;
  const alphaData = alpha?.data;
  for (let p = 0; p < size; p++) {
    const o = p * 4;
    out[o] = floatToByte(src[p] as number);
    out[o + 1] = floatToByte(src[size + p] as number);
    out[o + 2] = floatToByte(src[2 * size + p] as number);
    out[o + 3] = alphaData ? floatToByte(alphaData[p] as number) : 255;
  }
  return PNG.sync.write(png);
}

export async function savePng(path: string, rgb: Tensor, alpha?: Tensor): Promise<void> {
  await writeFile(path, encodePng(rgb, alpha));
}
