# Changelog

## 0.4.0 (2026-07-01)

### Performance

- SIMD-vectorized the WebAssembly convolution kernel: eight output pixels per iteration, one per `f64x2` lane, each still accumulating its taps in scalar order. End-to-end upscaling with the bundled models is ~3.7x faster, and outputs stay bit-identical to 0.3.x.
- Inference weights and biases are uploaded to Wasm memory once per model instead of once per block and layer.
- Raw conv parameters are extracted from the model JSON once per layer and cached; `runModel` no longer re-serializes networks on every block.
- brain.js networks are built lazily and only consulted for layers the fast engines cannot express.
- The pure-TypeScript convolution and deconvolution fallbacks clip kernel tap ranges up front instead of bounds-checking every tap, and pad/crop/block placement copy whole rows with `TypedArray.set`.

### Fixes

- `reconstructScale`/`scaleImage` with a factor other than the model's native one (for example 3, 4, or 1.5 with a 2x model) now runs the model repeatedly and Lanczos-resamples to the exact target instead of producing shifted or NaN output. Fractional factors are supported; output dimensions round to the nearest pixel.
- `crop` validates the region against the input bounds instead of silently reading out of range; `padEdge` rejects negative padding.
- Model JSON validation: conv layer network sizes must fit the declared kernel, deconv weight/bias counts must match the declared shape, and resize models must declare an integer `scaleFactor >= 2` with an `offset` that is a multiple of it.
- The Lanczos alpha path clamps its output to `[0, 1]` and always matches the RGB output dimensions.
- The test suite no longer fails when the optional brain.js dependency is absent; brain.js parity tests are skipped instead.

### CLI

- `--scale` accepts any factor > 0, fractional values included. Factors without a dedicated model file fall back to the bundled 2x model.
- Progress output reports the model file used and the actual engine (`gpu`, `wasm`, or `cpu`).

### Internal

- Removed dead code (`ensureAlphaPlane`, `isLastLayer`) and deduplicated `deconvOutputSize` between the engine and the trainer.
