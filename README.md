# kongyo2x

Image super-resolution in TypeScript. MLPconv networks, trained from scratch, upscale and denoise images — with the heavy convolutions, deconvolution, resampling, and training running in a bundled, SIMD-accelerated Rust/WebAssembly core (and a pure-TypeScript fallback that produces bit-identical output). No Python.

## Install

```sh
npm install @kongyo2/kongyo2x
```

Requires Node.js 20+. The heavy math runs in a prebuilt Rust/WebAssembly module that ships with the package — no toolchain or build step at install time, and a pure-TypeScript fallback covers any environment where the module can't load. GPU acceleration is still available: brain.js and its `gpu.js` peer compile a native module (`gl`, headless-gl) at install time, which needs X11/OpenGL system libraries and can't complete in a headless environment, so brain.js is an optional dependency — when its native build fails npm skips it and the install still succeeds.

## CLI

```sh
npx kongyo2x -i input.png -o output.png
```

| Option | Description | Default |
| --- | --- | --- |
| `-i, --input <path>` | input image (PNG or JPEG) | required |
| `-o, --output <path>` | output PNG path | `<name>_scale.png` |
| `-s, --scale <factor>` | upscale factor, any value > 0 | `2` |
| `--variant <name>` | model variant, e.g. `hq` | none |
| `-d, --model-dir <path>` | directory with `*_model.json` files | bundled models |
| `--block-size <n>` | tile size for processing | `128` |
| `--alpha-scale <mode>` | `model` or `lanczos` | `model` |
| `-q, --quiet` | suppress progress output | |

Two models ship with the package: the default `scale2.0x` and a higher-quality `scale2.0x_hq` (`--variant hq`). Factors without a dedicated model file — `3`, `4`, `1.5`, … — run the 2x model repeatedly and Lanczos-resample the result to the exact target size.

## Library

```ts
import { loadImage, savePng, loadModelFile, scaleImage } from "@kongyo2/kongyo2x";

const model = await loadModelFile("node_modules/@kongyo2/kongyo2x/models/mlpconv/scale2.0x_model.json");
const image = await loadImage("input.png");
const result = scaleImage(model, 2, image);
await savePng("output.png", result.rgb, result.alpha);
```

`scaleImage` accepts any factor > 0: a 2x model is applied as many times as needed and the result is resampled to the exact target when they differ.

## Environment variables

| Variable | Effect |
| --- | --- |
| `KONGYO2X_DISABLE_WASM=1` | skip the WebAssembly kernels and run the pure-TypeScript backend |
| `KONGYO2X_DISABLE_GPU=1` | skip the gpu.js probe and run on the CPU |

Both backends produce bit-identical output; the variables exist for debugging and benchmarking.

## Training

Training runs from a source checkout — `scripts/train.ts` is not part of the published npm package. Clone the repo, then train a model from scratch on your own images (or synthetic data if `--data` is omitted):

```sh
git clone https://github.com/kongyo2/kongyo2x
cd kongyo2x && npm install
npm run train -- --data ./images --scale 2 -o models/mlpconv/scale2.0x_model.json
```

The default `upconv` architecture matches the shipped models: valid 3x3 convolutions followed by a 4x4 stride-2 transposed convolution over RGB (`kongyo2x_upconv`, `resize: true`). Accuracy scales with `--iterations` and real `--data` — the defaults are a fast smoke run. Pass `--arch mlpconv` for the luma-only, nearest-pre-upscale variant. Run `npm run train -- --help` for all options.

## License

MIT
