import { readdir, readFile } from "node:fs/promises";
import { extname, join } from "node:path";
import type { Tensor } from "../core/tensor.js";
import { cloneTensor, createTensor } from "../core/tensor.js";
import { crop } from "../image/pad.js";
import { rgb2y } from "../image/color.js";
import { resizeLanczos, resizeNearest } from "../image/resize.js";
import { decodeImage } from "../image/io.js";
import type { Rng } from "./rng.js";

export interface Degradation {
  kind: "noise" | "scale";
  noiseSigma: number;
  scale: number;
}

const NOISE_SIGMA = [0.01, 0.03, 0.06, 0.1];

export function noiseSigmaForLevel(level: number): number {
  return NOISE_SIGMA[Math.max(0, Math.min(NOISE_SIGMA.length - 1, level))] as number;
}

function clamp01InPlace(data: Float32Array): void {
  for (let i = 0; i < data.length; i++) {
    const v = data[i] as number;
    data[i] = v < 0 ? 0 : v > 1 ? 1 : v;
  }
}

export function generateLuma(size: number, rng: Rng): Tensor {
  const plane = createTensor(1, size, size);
  const data = plane.data;
  const base = 0.2 + rng.next() * 0.6;
  const ax = (rng.next() * 2 - 1) * 0.6;
  const ay = (rng.next() * 2 - 1) * 0.6;
  const waves = 2 + Math.floor(rng.next() * 3);
  const waveParams: number[] = [];
  for (let w = 0; w < waves; w++) {
    waveParams.push(
      (0.5 + rng.next() * 6) / size,
      (0.5 + rng.next() * 6) / size,
      rng.next() * Math.PI * 2,
      (rng.next() * 0.25) / waves,
    );
  }
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let v = base + ax * (x / size - 0.5) + ay * (y / size - 0.5);
      for (let w = 0; w < waves; w++) {
        const fx = waveParams[w * 4] as number;
        const fy = waveParams[w * 4 + 1] as number;
        const phase = waveParams[w * 4 + 2] as number;
        const amp = waveParams[w * 4 + 3] as number;
        v += amp * Math.sin(2 * Math.PI * (fx * x + fy * y) + phase);
      }
      data[y * size + x] = v;
    }
  }

  const shapes = 2 + Math.floor(rng.next() * 4);
  for (let s = 0; s < shapes; s++) {
    const gray = rng.next();
    const cx = rng.next() * size;
    const cy = rng.next() * size;
    if (rng.next() < 0.5) {
      const radius = (0.08 + rng.next() * 0.22) * size;
      const r2 = radius * radius;
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          const dx = x - cx;
          const dy = y - cy;
          if (dx * dx + dy * dy <= r2) {
            data[y * size + x] = gray;
          }
        }
      }
    } else {
      const hw = (0.08 + rng.next() * 0.25) * size;
      const hh = (0.08 + rng.next() * 0.25) * size;
      const x0 = Math.max(0, Math.floor(cx - hw));
      const x1 = Math.min(size, Math.ceil(cx + hw));
      const y0 = Math.max(0, Math.floor(cy - hh));
      const y1 = Math.min(size, Math.ceil(cy + hh));
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          data[y * size + x] = gray;
        }
      }
    }
  }
  clamp01InPlace(data);
  return plane;
}

export function degrade(clean: Tensor, deg: Degradation, rng: Rng): Tensor {
  let input: Tensor;
  if (deg.kind === "scale") {
    const low = resizeLanczos(
      clean,
      Math.max(1, Math.round(clean.width / deg.scale)),
      Math.max(1, Math.round(clean.height / deg.scale)),
    );
    input = resizeNearest(low, clean.width, clean.height);
  } else {
    input = cloneTensor(clean);
  }
  if (deg.noiseSigma > 0) {
    const sigma = deg.noiseSigma * (0.7 + 0.3 * rng.next());
    const data = input.data;
    for (let i = 0; i < data.length; i++) {
      data[i] = (data[i] as number) + rng.gaussian() * sigma;
    }
    clamp01InPlace(data);
  }
  return input;
}

export interface Dataset {
  sample(size: number, rng: Rng): Tensor;
}

export class SyntheticDataset implements Dataset {
  sample(size: number, rng: Rng): Tensor {
    return generateLuma(size, rng);
  }
}

export class ImageDataset implements Dataset {
  private readonly planes: Tensor[];

  private constructor(planes: Tensor[]) {
    this.planes = planes;
  }

  static async load(dir: string, minSize: number): Promise<ImageDataset> {
    const entries = await readdir(dir);
    const planes: Tensor[] = [];
    for (const entry of entries) {
      const ext = extname(entry).toLowerCase();
      if (ext !== ".png" && ext !== ".jpg" && ext !== ".jpeg") {
        continue;
      }
      const buffer = await readFile(join(dir, entry));
      const image = decodeImage(buffer);
      const y = rgb2y(image.rgb);
      if (y.width >= minSize && y.height >= minSize) {
        planes.push(y);
      }
    }
    if (planes.length === 0) {
      throw new Error(`no usable images (>= ${minSize}px) found in ${dir}`);
    }
    return new ImageDataset(planes);
  }

  sample(size: number, rng: Rng): Tensor {
    const plane = this.planes[Math.floor(rng.next() * this.planes.length)] as Tensor;
    const x0 = Math.floor(rng.next() * (plane.width - size + 1));
    const y0 = Math.floor(rng.next() * (plane.height - size + 1));
    return crop(plane, x0, y0, x0 + size, y0 + size);
  }
}
