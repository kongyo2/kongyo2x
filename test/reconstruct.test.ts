import { describe, expect, it } from "vitest";
import { Kongyo2xModel } from "../src/model/model.js";
import { reconstructScale } from "../src/reconstruct.js";
import { scaleImage } from "../src/pipeline.js";
import type { ConvLayerJSON, DeconvLayerJSON } from "../src/model/types.js";
import { vggLikeModel, upconvLikeModel, randomImage, maxAbsDiff, hasNaN } from "./helpers.js";

describe("model metadata", () => {
  it("reads metadata from a vgg model", () => {
    const model = Kongyo2xModel.fromJSON(vggLikeModel(1));
    expect(model.meta.archName).toBe("vgg_7");
    expect(model.meta.channels).toBe(1);
    expect(model.meta.scaleFactor).toBe(1);
    expect(model.meta.offset).toBe(3);
    expect(model.isRgb).toBe(false);
  });

  it("recognizes an upconv (deconv) model as 2x", () => {
    const model = Kongyo2xModel.fromJSON(upconvLikeModel(3));
    expect(model.meta.scaleFactor).toBe(2);
    expect(model.meta.resize).toBe(true);
    expect(model.isRgb).toBe(true);
  });
});

describe("model validation", () => {
  it("rejects conv layers whose network sizes do not fit the kernel", () => {
    const json = vggLikeModel(1);
    (json.layers[0] as ConvLayerJSON).network.sizes = [10, 4];
    expect(() => Kongyo2xModel.fromJSON(json)).toThrow(/do not fit/);
  });

  it("rejects deconv layers with mismatched weight counts", () => {
    const json = upconvLikeModel(3);
    const deconv = json.layers[2] as DeconvLayerJSON;
    deconv.weights = deconv.weights.slice(0, 10);
    expect(() => Kongyo2xModel.fromJSON(json)).toThrow(/deconv layer expects/);
  });

  it("extracts conv parameters once per layer and caches them", () => {
    const model = Kongyo2xModel.fromJSON(vggLikeModel(1));
    const first = model.convParams(0);
    expect(first).toBeDefined();
    expect(model.convParams(0)).toBe(first);
  });
});

describe("model JSON round-trip", () => {
  it("re-serializes to a model that produces identical output", () => {
    const model = Kongyo2xModel.fromJSON(vggLikeModel(1));
    const round = Kongyo2xModel.fromJSON(model.toJSON());
    const image = randomImage(3, 24, 24, 5);
    const a = reconstructScale(model, 2, image, { blockSize: 32 });
    const b = reconstructScale(round, 2, image, { blockSize: 32 });
    expect(maxAbsDiff(a, b)).toBe(0);
  });

  it("preserves the deconv layer through a round-trip", () => {
    const model = Kongyo2xModel.fromJSON(upconvLikeModel(3));
    const round = Kongyo2xModel.fromJSON(model.toJSON());
    const image = randomImage(3, 16, 16, 7);
    const a = scaleImage(model, 2, { rgb: image }, { blockSize: 24 });
    const b = scaleImage(round, 2, { rgb: image }, { blockSize: 24 });
    expect(maxAbsDiff(a.rgb, b.rgb)).toBe(0);
  });
});

describe("block-size invariance", () => {
  it("gives identical results for different tile sizes", () => {
    const model = Kongyo2xModel.fromJSON(vggLikeModel(1));
    const image = randomImage(3, 30, 30, 9);
    const small = reconstructScale(model, 2, image, { blockSize: 20 });
    const large = reconstructScale(model, 2, image, { blockSize: 40 });
    expect(maxAbsDiff(small, large)).toBeLessThan(1e-6);
  });
});

describe("scaling", () => {
  it("doubles image dimensions with an upconv model", () => {
    const model = Kongyo2xModel.fromJSON(upconvLikeModel(3));
    const image = { rgb: randomImage(3, 16, 18, 5) };
    const result = scaleImage(model, 2, image, { blockSize: 24 });
    expect(result.rgb.width).toBe(36);
    expect(result.rgb.height).toBe(32);
  });

  it("doubles image dimensions with a vgg (pre-upscale) model", () => {
    const model = Kongyo2xModel.fromJSON(vggLikeModel(1));
    const image = { rgb: randomImage(3, 12, 14, 4) };
    const result = reconstructScale(model, 2, image.rgb, { blockSize: 24 });
    expect(result.width).toBe(28);
    expect(result.height).toBe(24);
  });

  it("preserves and upscales alpha through the scale pipeline", () => {
    const model = Kongyo2xModel.fromJSON(upconvLikeModel(3));
    const rgb = randomImage(3, 16, 16, 1);
    const alpha = randomImage(1, 16, 16, 2);
    const result = scaleImage(model, 2, { rgb, alpha }, { blockSize: 24, alphaScale: "lanczos" });
    expect(result.alpha?.width).toBe(32);
    expect(result.alpha?.height).toBe(32);
  });
});

describe("arbitrary scale factors", () => {
  it("reaches 4x by running a 2x model twice", () => {
    const model = Kongyo2xModel.fromJSON(upconvLikeModel(3));
    const image = randomImage(3, 12, 10, 3);
    const out = reconstructScale(model, 4, image, { blockSize: 20 });
    expect([out.width, out.height]).toEqual([40, 48]);
    expect(hasNaN(out)).toBe(false);
  });

  it("reaches 3x via a second pass and a Lanczos downsample", () => {
    const model = Kongyo2xModel.fromJSON(upconvLikeModel(3));
    const image = randomImage(3, 12, 10, 4);
    const out = reconstructScale(model, 3, image, { blockSize: 20 });
    expect([out.width, out.height]).toEqual([30, 36]);
    expect(hasNaN(out)).toBe(false);
  });

  it("supports scale 1 with a resize model (refine, then downsample)", () => {
    const model = Kongyo2xModel.fromJSON(upconvLikeModel(3));
    const image = randomImage(3, 12, 10, 5);
    const out = reconstructScale(model, 1, image, { blockSize: 20 });
    expect([out.width, out.height]).toEqual([10, 12]);
    expect(hasNaN(out)).toBe(false);
  });

  it("supports fractional scales with rounded output dimensions", () => {
    const model = Kongyo2xModel.fromJSON(upconvLikeModel(3));
    const image = randomImage(3, 12, 10, 6);
    const out = reconstructScale(model, 1.5, image, { blockSize: 20 });
    expect([out.width, out.height]).toEqual([15, 18]);
    expect(hasNaN(out)).toBe(false);
  });

  it("supports 3x with a pre-upscale (vgg) model", () => {
    const model = Kongyo2xModel.fromJSON(vggLikeModel(1));
    const image = randomImage(3, 8, 10, 7);
    const out = reconstructScale(model, 3, image, { blockSize: 24 });
    expect([out.width, out.height]).toEqual([30, 24]);
    expect(hasNaN(out)).toBe(false);
  });

  it("rejects non-positive and non-finite scale factors", () => {
    const model = Kongyo2xModel.fromJSON(upconvLikeModel(3));
    const image = randomImage(3, 8, 8, 8);
    expect(() => reconstructScale(model, 0, image)).toThrow(/invalid scale/);
    expect(() => reconstructScale(model, Number.NaN, image)).toThrow(/invalid scale/);
  });

  it("keeps rgb and alpha dimensions in sync at 4x", () => {
    const model = Kongyo2xModel.fromJSON(upconvLikeModel(3));
    const rgb = randomImage(3, 10, 10, 1);
    const alpha = randomImage(1, 10, 10, 2);
    const result = scaleImage(model, 4, { rgb, alpha }, { blockSize: 20, alphaScale: "lanczos" });
    expect([result.rgb.width, result.rgb.height]).toEqual([40, 40]);
    expect([result.alpha?.width, result.alpha?.height]).toEqual([40, 40]);
  });
});
