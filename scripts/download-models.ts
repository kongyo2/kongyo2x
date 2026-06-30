import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";

const BASE = "https://raw.githubusercontent.com/nagadomi/waifu2x/master/models";

const CANDIDATE_FILES = [
  "noise0_model.json",
  "noise1_model.json",
  "noise2_model.json",
  "noise3_model.json",
  "scale2.0x_model.json",
  "noise0_scale2.0x_model.json",
  "noise1_scale2.0x_model.json",
  "noise2_scale2.0x_model.json",
  "noise3_scale2.0x_model.json",
];

const DEFAULT_DIRS = ["vgg_7/art_y", "vgg_7/art", "upconv_7/art"];

async function download(dir: string): Promise<void> {
  for (const file of CANDIDATE_FILES) {
    const url = `${BASE}/${dir}/${file}`;
    const response = await fetch(url, { headers: { "User-Agent": "waifu2x-brainjs/0.1" } });
    if (!response.ok) {
      continue;
    }
    const target = join("models", dir, file);
    await mkdir(dirname(target), { recursive: true });
    const buffer = Buffer.from(await response.arrayBuffer());
    await writeFile(target, buffer);
    process.stdout.write(`downloaded ${target} (${(buffer.length / 1024).toFixed(0)} KB)\n`);
  }
}

async function main(): Promise<void> {
  const dirs = process.argv.slice(2);
  const targets = dirs.length > 0 ? dirs : DEFAULT_DIRS;
  for (const dir of targets) {
    process.stdout.write(`fetching model set: ${dir}\n`);
    await download(dir);
  }
}

main().catch((error: unknown) => {
  process.stderr.write(`error: ${error instanceof Error ? error.message : String(error)}\n`);
  process.exit(1);
});
