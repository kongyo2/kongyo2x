#!/usr/bin/env node
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { parseArgs } from "node:util";
import { defaultTrainConfig, trainModel, SyntheticDataset, ImageDataset } from "../src/train/index.js";
import type { Dataset, TrainArch, TrainConfig, TrainLoss, TrainProgress } from "../src/train/index.js";

interface TrainCliOptions {
  scale: number;
  dataDir: string | undefined;
  output: string;
  channels: number;
  config: TrainConfig;
  quiet: boolean;
}

const HELP = `kongyo2x train - train a super-resolution model from scratch

Usage:
  npm run train -- [options]

Options:
  -s, --scale <factor>      upscale factor                        (default: 2)
      --arch <name>         upconv | mlpconv                      (default: upconv)
      --data <dir>          directory of training images (PNG/JPEG); synthetic if omitted
  -o, --output <path>       output model JSON path
  -d, --model-dir <path>    output directory                      (default: ./models/mlpconv)
      --channels <n>        1 (luma) or 3 (rgb)                   (default: 3 upconv, 1 mlpconv)
      --conv-channels <l>   upconv body widths, comma separated   (default: 16,32,64,128,128,256)
      --iterations <n>      training iterations                   (default: 600)
      --batch <n>           patches per iteration                 (default: 8)
      --patch <n>           training patch size (hi-res)          (default: 64 upconv, 40 mlpconv)
      --width <n>           mlpconv channel width                 (default: 20)
      --lr <rate>           base Adam learning rate               (default: 2e-4 upconv, 1e-3 mlpconv)
      --loss <kind>         charbonnier | mse | mix               (default: mix upconv, charbonnier mlpconv)
      --seed <n>            random seed                           (default: 1234)
  -q, --quiet               suppress progress output
  -h, --help                show this help

The upconv architecture reproduces the shipped models: valid 3x3 convs + a 4x4 stride-2
transposed convolution (kongyo2x_upconv, resize=true). Quality scales with --iterations and
real --data; the defaults are a fast smoke run.
`;

function parseInteger(value: string, name: string, min: number): number {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isInteger(parsed) || parsed < min) {
    throw new Error(`invalid ${name}: ${value}`);
  }
  return parsed;
}

function parseConvChannels(value: string): number[] {
  const widths = value
    .split(",")
    .map((part) => part.trim())
    .filter((part) => part.length > 0)
    .map((part) => parseInteger(part, "conv-channels", 1));
  if (widths.length === 0) {
    throw new Error(`invalid conv-channels: ${value}`);
  }
  return widths;
}

function parse(argv: string[]): TrainCliOptions {
  const { values } = parseArgs({
    args: argv,
    options: {
      scale: { type: "string", short: "s", default: "2" },
      arch: { type: "string", default: "upconv" },
      data: { type: "string" },
      output: { type: "string", short: "o" },
      "model-dir": { type: "string", short: "d", default: "./models/mlpconv" },
      channels: { type: "string" },
      "conv-channels": { type: "string" },
      iterations: { type: "string", default: "600" },
      batch: { type: "string", default: "8" },
      patch: { type: "string" },
      width: { type: "string" },
      lr: { type: "string" },
      loss: { type: "string" },
      seed: { type: "string", default: "1234" },
      quiet: { type: "boolean", short: "q", default: false },
      help: { type: "boolean", short: "h", default: false },
    },
    allowPositionals: false,
  });

  if (values.help) {
    process.stdout.write(HELP);
    process.exit(0);
  }

  const arch = values.arch as string;
  if (arch !== "upconv" && arch !== "mlpconv") {
    throw new Error(`invalid arch: ${arch} (expected upconv|mlpconv)`);
  }
  const scale = parseInteger(values.scale as string, "scale", 1);
  const channels =
    values.channels !== undefined ? parseInteger(values.channels as string, "channels", 1) : arch === "upconv" ? 3 : 1;

  const overrides: Partial<TrainConfig> = {
    arch: arch as TrainArch,
    channels,
    degradation: { scale },
    iterations: parseInteger(values.iterations as string, "iterations", 1),
    batchSize: parseInteger(values.batch as string, "batch", 1),
    seed: parseInteger(values.seed as string, "seed", 0),
  };
  if (values.patch !== undefined) {
    overrides.patchSize = parseInteger(values.patch as string, "patch", 8);
  }
  if (values.width !== undefined) {
    overrides.width = parseInteger(values.width as string, "width", 2);
  }
  if (values["conv-channels"] !== undefined) {
    overrides.convChannels = parseConvChannels(values["conv-channels"] as string);
  }
  if (values.loss !== undefined) {
    const loss = values.loss as TrainLoss;
    if (loss !== "charbonnier" && loss !== "mse" && loss !== "mix") {
      throw new Error(`invalid loss: ${values.loss} (expected charbonnier|mse|mix)`);
    }
    overrides.loss = loss;
  }
  if (values.lr !== undefined) {
    const lr = Number.parseFloat(values.lr as string);
    if (!Number.isFinite(lr) || lr <= 0) {
      throw new Error(`invalid lr: ${values.lr}`);
    }
    overrides.baseLr = lr;
  }

  const config = defaultTrainConfig(overrides);
  const modelDir = values["model-dir"] as string;
  const output = values.output ?? join(modelDir, `scale${scale.toFixed(1)}x_model.json`);

  return {
    scale,
    dataDir: values.data,
    output,
    channels,
    config,
    quiet: values.quiet ?? false,
  };
}

async function buildDataset(options: TrainCliOptions): Promise<Dataset> {
  if (options.dataDir) {
    return ImageDataset.load(options.dataDir, options.config.patchSize, options.channels);
  }
  return new SyntheticDataset(options.channels);
}

async function main(): Promise<void> {
  const options = parse(process.argv.slice(2));
  const log = (message: string): void => {
    if (!options.quiet) {
      process.stderr.write(`${message}\n`);
    }
  };

  const dataset = await buildDataset(options);
  const source = options.dataDir ?? "synthetic";
  const cfg = options.config;
  log(
    `training scale ${options.scale}x ${cfg.arch} model (${source} data, ${cfg.channels}ch, ` +
      `${cfg.iterations} iters, batch ${cfg.batchSize}, patch ${cfg.patchSize}, loss ${cfg.loss})`,
  );

  const onProgress = (p: TrainProgress): void => {
    log(
      `step ${p.step}  loss ${p.loss.toExponential(3)}  lr ${p.lr.toExponential(2)}  ` +
        `|g| ${p.gradNorm.toFixed(3)}  val ${p.valPsnr.toFixed(2)}dB  (baseline ${p.baselinePsnr.toFixed(2)}dB)`,
    );
  };

  const start = performance.now();
  const result = trainModel(cfg, dataset, onProgress);
  const seconds = ((performance.now() - start) / 1000).toFixed(1);

  const json = JSON.stringify(result.network.toModelJSON(result.meta));
  await mkdir(dirname(options.output), { recursive: true });
  await writeFile(options.output, json);

  const gain = result.finalValPsnr - result.baselinePsnr;
  log(
    `done in ${seconds}s  final ${result.finalValPsnr.toFixed(2)}dB  ` +
      `(start ${result.initialValPsnr.toFixed(2)}dB, baseline ${result.baselinePsnr.toFixed(2)}dB)  ` +
      `gain vs baseline ${gain >= 0 ? "+" : ""}${gain.toFixed(2)}dB${result.usedEma ? "  (ema)" : ""}`,
  );
  process.stdout.write(`wrote ${options.output} (${(json.length / 1024).toFixed(0)} KB)\n`);
}

main().catch((error: unknown) => {
  process.stderr.write(`error: ${error instanceof Error ? error.message : String(error)}\n`);
  process.exit(1);
});
