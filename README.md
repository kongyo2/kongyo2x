# kongyo2x

Image super-resolution in TypeScript. MLPconv networks, trained from scratch and run through [brain.js](https://brain.js.org/), upscale and denoise images — pure TypeScript, no Python.

## Install

```sh
npm install @kongyo2/kongyo2x
```

Requires Node.js 20+. Installation pulls a transitive native module (`gl`, via brain.js's `gpu.js` peer dependency) that builds at install time to enable optional GPU acceleration. If that build fails — or you install with `--ignore-scripts` — kongyo2x falls back to brain.js on the CPU.

## CLI

```sh
npx kongyo2x -i input.png -o output.png
```

| Option | Description | Default |
| --- | --- | --- |
| `-i, --input <path>` | input image (PNG or JPEG) | required |
| `-o, --output <path>` | output PNG path | `<name>_scale.png` |
| `-s, --scale <factor>` | upscale factor | `2` |
| `--variant <name>` | model variant, e.g. `hq` | none |
| `--block-size <n>` | tile size for processing | `128` |
| `--alpha-scale <mode>` | `model` or `lanczos` | `model` |
| `-q, --quiet` | suppress progress output | |

Two models ship with the package: the default `scale2.0x` and a higher-quality `scale2.0x_hq` (`--variant hq`).

## Library

```ts
import { loadImage, savePng, loadModelFile, scaleImage } from "@kongyo2/kongyo2x";

const model = await loadModelFile("node_modules/@kongyo2/kongyo2x/models/mlpconv/scale2.0x_model.json");
const image = await loadImage("input.png");
const result = scaleImage(model, 2, image);
await savePng("output.png", result.rgb, result.alpha);
```

## Training

Training runs from a source checkout — `scripts/train.ts` is not part of the published npm package. Clone the repo, then train a model from scratch on your own images (or synthetic data if `--data` is omitted):

```sh
git clone https://github.com/kongyo2/kongyo2x
cd kongyo2x && npm install
npm run train -- --data ./images --scale 2 -o models/mlpconv/scale2.0x_model.json
```

Run `npm run train -- --help` for all options.

## License

MIT
