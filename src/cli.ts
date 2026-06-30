#!/usr/bin/env node
import { parseArgs } from "node:util";
import { fileURLToPath } from "node:url";
import { basename, dirname, extname, join } from "node:path";
import { loadImage, savePng } from "./image/io.js";
import { loadModelFile, scaleImage } from "./pipeline.js";
import type { ScaleOptions } from "./pipeline.js";
import { isGpuAvailable } from "./brain/loader.js";

const BUNDLED_MODEL_DIR = fileURLToPath(new URL("../models/mlpconv", import.meta.url));

interface CliOptions {
  input: string;
  output: string;
  scale: number;
  variant: string;
  modelDir: string;
  blockSize: number;
  alphaScale: "model" | "lanczos";
  quiet: boolean;
}

const HELP = `kongyo2x - image super-resolution powered by brain.js

Usage:
  kongyo2x -i input.png [options]

Options:
  -i, --input <path>        input image (PNG or JPEG)            [required]
  -o, --output <path>       output PNG path (default: <name>_scale.png)
  -s, --scale <factor>      upscale factor                       (default: 2)
      --variant <name>      model variant, e.g. hq               (default: none)
  -d, --model-dir <path>    directory with *_model.json files    (default: bundled models)
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
      scale: { type: "string", short: "s", default: "2" },
      variant: { type: "string", default: "" },
      "model-dir": { type: "string", short: "d", default: BUNDLED_MODEL_DIR },
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

  const alphaScale = values["alpha-scale"] as "model" | "lanczos";
  if (alphaScale !== "model" && alphaScale !== "lanczos") {
    throw new Error(`invalid alpha-scale: ${values["alpha-scale"]} (expected model|lanczos)`);
  }
  const scale = Number.parseInt(values.scale as string, 10);
  if (!Number.isInteger(scale) || scale < 1) {
    throw new Error(`invalid scale: ${values.scale}`);
  }
  const variant = values.variant as string;
  if (variant !== "" && !/^[a-z0-9_-]+$/i.test(variant)) {
    throw new Error(`invalid variant: ${variant} (expected letters, digits, '-' or '_')`);
  }
  const blockSize = Number.parseInt(values["block-size"] as string, 10);
  if (!Number.isInteger(blockSize) || blockSize < 16) {
    throw new Error(`invalid block size: ${values["block-size"]}`);
  }

  const input = values.input;
  const defaultOutput = join(dirname(input), `${basename(input, extname(input))}_scale.png`);
  return {
    input,
    output: values.output ?? defaultOutput,
    scale,
    variant,
    modelDir: values["model-dir"] as string,
    blockSize,
    alphaScale,
    quiet: values.quiet ?? false,
  };
}

function scaleTag(scale: number): string {
  return `${scale.toFixed(1)}x`;
}

function modelFileName(scale: number, variant: string): string {
  const tag = scaleTag(scale);
  return variant ? `scale${tag}_${variant}_model.json` : `scale${tag}_model.json`;
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

  const start = performance.now();
  const model = await loadModelFile(join(options.modelDir, modelFileName(options.scale, options.variant)));
  log(
    `scale ${options.scale}x${options.variant ? ` (${options.variant})` : ""} · engine: ${isGpuAvailable() ? "gpu" : "cpu"}`,
  );
  const result = scaleImage(model, options.scale, image, scaleOptions);

  await savePng(options.output, result.rgb, result.alpha);
  const seconds = ((performance.now() - start) / 1000).toFixed(2);
  log(`wrote ${options.output} (${seconds}s)`);
}

run(parse(process.argv.slice(2))).catch((error: unknown) => {
  process.stderr.write(`error: ${error instanceof Error ? error.message : String(error)}\n`);
  process.exit(1);
});
