import { describe, expect, it } from "vitest";
import { Kongyo2xModel } from "../src/model/model.js";
import { reconstructImage, reconstructScale } from "../src/reconstruct.js";
import { denoise, scaleImage } from "../src/pipeline.js";
import { vggLikeModel, upconvLikeModel, randomImage, maxAbsDiff } from "./helpers.js";

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

describe("model JSON round-trip", () => {
  it("re-serializes to a model that produces identical output", () => {
    const model = Kongyo2xModel.fromJSON(vggLikeModel(1));
    const round = Kongyo2xModel.fromJSON(model.toJSON());
    const image = randomImage(3, 24, 24, 5);
    const a = reconstructImage(model, image, { blockSize: 32 });
    const b = reconstructImage(round, image, { blockSize: 32 });
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

describe("denoise", () => {
  it("keeps the image dimensions", () => {
    const model = Kongyo2xModel.fromJSON(vggLikeModel(1));
    const image = { rgb: randomImage(3, 24, 28, 21) };
    const out = denoise(model, image, { blockSize: 32 });
    expect(out.rgb.width).toBe(28);
    expect(out.rgb.height).toBe(24);
  });
});

describe("block-size invariance", () => {
  it("gives identical results for different tile sizes", () => {
    const model = Kongyo2xModel.fromJSON(vggLikeModel(1));
    const image = randomImage(3, 30, 30, 9);
    const small = reconstructImage(model, image, { blockSize: 20 });
    const large = reconstructImage(model, image, { blockSize: 40 });
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
