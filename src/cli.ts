#!/usr/bin/env node
import { parseArgs } from "node:util";
import { existsSync } from "node:fs";
import { basename, dirname, extname, join } from "node:path";
import { loadImage, savePng } from "./image/io.js";
import { loadModelFile, denoise, scaleImage, denoiseThenScale } from "./pipeline.js";
import type { ProcessedImage, ScaleOptions } from "./pipeline.js";

type Method = "noise" | "scale" | "noise_scale";

interface CliOptions {
  input: string;
  output: string;
  method: Method;
  noiseLevel: number;
  scale: number;
  modelDir: string;
  blockSize: number;
  alphaScale: "model" | "lanczos";
  quiet: boolean;
}

const HELP = `kongyo2x - image super-resolution and noise reduction powered by brain.js

Usage:
  kongyo2x -i input.png [options]

Options:
  -i, --input <path>        input image (PNG or JPEG)            [required]
  -o, --output <path>       output PNG path (default: <name>_<method>.png)
  -m, --method <method>     noise | scale | noise_scale          (default: noise_scale)
  -n, --noise <level>       denoise level 0-3                    (default: 1)
  -s, --scale <factor>      upscale factor                       (default: 2)
  -d, --model-dir <path>    directory with *_model.json files    (default: ./models/vgg_7/art)
      --block-size <n>      tile size for processing             (default: 128)
      --alpha-scale <mode>  model | lanczos (alpha upscaling)    (default: model)
  -q, --quiet               suppress progress output
  -h, --help                show this help
`;

function parse(argv: string[]): CliOptions {
  const { values } = parseArgs({
    args: argv,
    options: {
      input: { type: "string", short: "i" },
      output: { type: "string", short: "o" },
      method: { type: "string", short: "m", default: "noise_scale" },
      noise: { type: "string", short: "n", default: "1" },
      scale: { type: "string", short: "s", default: "2" },
      "model-dir": { type: "string", short: "d", default: "./models/vgg_7/art" },
      "block-size": { type: "string", default: "128" },
      "alpha-scale": { type: "string", default: "model" },
      quiet: { type: "boolean", short: "q", default: false },
      help: { type: "boolean", short: "h", default: false },
    },
    allowPositionals: false,
  });

  if (values.help || !values.input) {
    process.stdout.write(HELP);
    process.exit(values.help ? 0 : 1);
  }

  const method = values.method as Method;
  if (method !== "noise" && method !== "scale" && method !== "noise_scale") {
    throw new Error(`invalid method: ${values.method} (expected noise|scale|noise_scale)`);
  }
  const alphaScale = values["alpha-scale"] as "model" | "lanczos";
  if (alphaScale !== "model" && alphaScale !== "lanczos") {
    throw new Error(`invalid alpha-scale: ${values["alpha-scale"]} (expected model|lanczos)`);
  }
  const noiseLevel = Number.parseInt(values.noise as string, 10);
  if (!Number.isInteger(noiseLevel) || noiseLevel < 0 || noiseLevel > 3) {
    throw new Error(`invalid noise level: ${values.noise} (expected 0-3)`);
  }
  const scale = Number.parseInt(values.scale as string, 10);
  if (!Number.isInteger(scale) || scale < 1) {
    throw new Error(`invalid scale: ${values.scale}`);
  }
  const blockSize = Number.parseInt(values["block-size"] as string, 10);
  if (!Number.isInteger(blockSize) || blockSize < 16) {
    throw new Error(`invalid block size: ${values["block-size"]}`);
  }

  const input = values.input;
  const defaultOutput = join(dirname(input), `${basename(input, extname(input))}_${method}.png`);
  return {
    input,
    output: values.output ?? defaultOutput,
    method,
    noiseLevel,
    scale,
    modelDir: values["model-dir"] as string,
    blockSize,
    alphaScale,
    quiet: values.quiet ?? false,
  };
}

function scaleTag(scale: number): string {
  return `${scale.toFixed(1)}x`;
}

async function run(options: CliOptions): Promise<void> {
  const image = await loadImage(options.input);
  const scaleOptions: ScaleOptions = {
    blockSize: options.blockSize,
    alphaScale: options.alphaScale,
  };
  const log = (message: string): void => {
    if (!options.quiet) {
      process.stderr.write(`${message}\n`);
    }
  };

  let result: ProcessedImage;
  const start = performance.now();

  if (options.method === "noise") {
    const model = await loadModelFile(join(options.modelDir, `noise${options.noiseLevel}_model.json`));
    log(`denoise (level ${options.noiseLevel})`);
    result = denoise(model, image, { blockSize: options.blockSize });
  } else if (options.method === "scale") {
    const model = await loadModelFile(join(options.modelDir, `scale${scaleTag(options.scale)}_model.json`));
    log(`scale ${options.scale}x`);
    result = scaleImage(model, options.scale, image, scaleOptions);
  } else {
    const combinedPath = join(
      options.modelDir,
      `noise${options.noiseLevel}_scale${scaleTag(options.scale)}_model.json`,
    );
    if (existsSync(combinedPath)) {
      const model = await loadModelFile(combinedPath);
      const scaleOnlyPath = join(options.modelDir, `scale${scaleTag(options.scale)}_model.json`);
      const alphaModel = image.alpha && existsSync(scaleOnlyPath) ? await loadModelFile(scaleOnlyPath) : model;
      log(`denoise+scale (combined model)`);
      result = scaleImage(model, options.scale, image, scaleOptions, alphaModel);
    } else {
      const noiseModel = await loadModelFile(join(options.modelDir, `noise${options.noiseLevel}_model.json`));
      const scaleModel = await loadModelFile(join(options.modelDir, `scale${scaleTag(options.scale)}_model.json`));
      log(`denoise (level ${options.noiseLevel}) then scale ${options.scale}x`);
      result = denoiseThenScale(noiseModel, scaleModel, options.scale, image, scaleOptions);
    }
  }

  await savePng(options.output, result.rgb, result.alpha);
  const seconds = ((performance.now() - start) / 1000).toFixed(2);
  log(`wrote ${options.output} (${seconds}s)`);
}

run(parse(process.argv.slice(2))).catch((error: unknown) => {
  process.stderr.write(`error: ${error instanceof Error ? error.message : String(error)}\n`);
  process.exit(1);
});
