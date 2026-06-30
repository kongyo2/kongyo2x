#!/usr/bin/env node
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { parseArgs } from "node:util";
import { defaultTrainConfig, trainModel, SyntheticDataset, ImageDataset } from "../src/train/index.js";
import type { Dataset, LossKind, TrainConfig, TrainProgress } from "../src/train/index.js";

interface TrainCliOptions {
  scale: number;
  dataDir: string | undefined;
  output: string;
  config: TrainConfig;
  quiet: boolean;
}

const HELP = `kongyo2x train - train an MLPconv super-resolution model from scratch

Usage:
  npm run train -- [options]

Options:
  -s, --scale <factor>      upscale factor                        (default: 2)
      --data <dir>          directory of training images (PNG/JPEG); synthetic if omitted
  -o, --output <path>       output model JSON path
  -d, --model-dir <path>    output directory                      (default: ./models/mlpconv)
      --iterations <n>      training iterations                   (default: 600)
      --batch <n>           patches per iteration                 (default: 8)
      --patch <n>           training patch size                   (default: 40)
      --width <n>           network channel width                 (default: 20)
      --lr <rate>           base Adam learning rate               (default: 0.001)
      --loss <kind>         charbonnier | mse                     (default: charbonnier)
      --seed <n>            random seed                           (default: 1234)
  -q, --quiet               suppress progress output
  -h, --help                show this help
`;

function parseInteger(value: string, name: string, min: number): number {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isInteger(parsed) || parsed < min) {
    throw new Error(`invalid ${name}: ${value}`);
  }
  return parsed;
}

function parse(argv: string[]): TrainCliOptions {
  const { values } = parseArgs({
    args: argv,
    options: {
      scale: { type: "string", short: "s", default: "2" },
      data: { type: "string" },
      output: { type: "string", short: "o" },
      "model-dir": { type: "string", short: "d", default: "./models/mlpconv" },
      iterations: { type: "string", default: "600" },
      batch: { type: "string", default: "8" },
      patch: { type: "string", default: "40" },
      width: { type: "string", default: "20" },
      lr: { type: "string", default: "0.001" },
      loss: { type: "string", default: "charbonnier" },
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

  const loss = values.loss as LossKind;
  if (loss !== "charbonnier" && loss !== "mse") {
    throw new Error(`invalid loss: ${values.loss} (expected charbonnier|mse)`);
  }
  const scale = parseInteger(values.scale as string, "scale", 1);
  const lr = Number.parseFloat(values.lr as string);
  if (!Number.isFinite(lr) || lr <= 0) {
    throw new Error(`invalid lr: ${values.lr}`);
  }

  const config = defaultTrainConfig({
    degradation: { scale },
    iterations: parseInteger(values.iterations as string, "iterations", 1),
    batchSize: parseInteger(values.batch as string, "batch", 1),
    patchSize: parseInteger(values.patch as string, "patch", 8),
    width: parseInteger(values.width as string, "width", 2),
    baseLr: lr,
    loss,
    seed: parseInteger(values.seed as string, "seed", 0),
  });

  const modelDir = values["model-dir"] as string;
  const output = values.output ?? join(modelDir, `scale${scale.toFixed(1)}x_model.json`);

  return {
    scale,
    dataDir: values.data,
    output,
    config,
    quiet: values.quiet ?? false,
  };
}

async function buildDataset(options: TrainCliOptions): Promise<Dataset> {
  if (options.dataDir) {
    return ImageDataset.load(options.dataDir, options.config.patchSize);
  }
  return new SyntheticDataset();
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
  log(
    `training scale ${options.scale}x model (${source} data, ${options.config.iterations} iters, ` +
      `batch ${options.config.batchSize}, patch ${options.config.patchSize}, width ${options.config.width})`,
  );

  const onProgress = (p: TrainProgress): void => {
    log(
      `step ${p.step}  loss ${p.loss.toExponential(3)}  lr ${p.lr.toExponential(2)}  ` +
        `|g| ${p.gradNorm.toFixed(3)}  val ${p.valPsnr.toFixed(2)}dB  (baseline ${p.baselinePsnr.toFixed(2)}dB)`,
    );
  };

  const start = performance.now();
  const result = trainModel(options.config, dataset, onProgress);
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
