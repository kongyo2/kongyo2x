import { describe, expect, it } from "vitest";
import { Waifu2xModel } from "../src/model/model.js";
import { reconstructImage, reconstructScale } from "../src/reconstruct.js";
import { denoise, scaleImage } from "../src/pipeline.js";
import { vggLikeModel, upconvLikeModel, randomImage, maxAbsDiff } from "./helpers.js";

describe("model metadata", () => {
  it("reads metadata from model_config", () => {
    const model = Waifu2xModel.fromJSON(vggLikeModel(1));
    expect(model.meta.archName).toBe("vgg_7");
    expect(model.meta.channels).toBe(1);
    expect(model.meta.scaleFactor).toBe(1);
    expect(model.meta.offset).toBe(3);
    expect(model.isRgb).toBe(false);
  });

  it("recognizes an upconv (deconv) model as 2x", () => {
    const model = Waifu2xModel.fromJSON(upconvLikeModel(3));
    expect(model.meta.scaleFactor).toBe(2);
    expect(model.meta.resize).toBe(true);
    expect(model.isRgb).toBe(true);
  });
});

describe("brain vs fast engine equivalence", () => {
  it("produces near-identical output for a Y-channel model (image)", () => {
    const model = Waifu2xModel.fromJSON(vggLikeModel(1));
    const image = { rgb: randomImage(3, 24, 28, 21) };
    const fast = denoise(model, image, { engine: "fast", blockSize: 32 });
    const brain = denoise(model, image, { engine: "brain", blockSize: 32 });
    expect(maxAbsDiff(fast.rgb, brain.rgb)).toBeLessThan(1e-3);
  });

  it("produces near-identical output for an RGB model (image)", () => {
    const model = Waifu2xModel.fromJSON(vggLikeModel(3));
    const image = randomImage(3, 20, 20, 33);
    const fast = reconstructImage(model, image, { engine: "fast", blockSize: 24 });
    const brain = reconstructImage(model, image, { engine: "brain", blockSize: 24 });
    expect(maxAbsDiff(fast, brain)).toBeLessThan(1e-3);
  });
});

describe("block-size invariance", () => {
  it("gives identical results for different tile sizes", () => {
    const model = Waifu2xModel.fromJSON(vggLikeModel(1));
    const image = randomImage(3, 30, 30, 9);
    const small = reconstructImage(model, image, { engine: "fast", blockSize: 20 });
    const large = reconstructImage(model, image, { engine: "fast", blockSize: 40 });
    expect(maxAbsDiff(small, large)).toBeLessThan(1e-6);
  });
});

describe("scaling", () => {
  it("doubles image dimensions with an upconv model", () => {
    const model = Waifu2xModel.fromJSON(upconvLikeModel(3));
    const image = { rgb: randomImage(3, 16, 18, 5) };
    const result = scaleImage(model, 2, image, { engine: "fast", blockSize: 24 });
    expect(result.rgb.width).toBe(36);
    expect(result.rgb.height).toBe(32);
  });

  it("doubles image dimensions with a vgg (pre-upscale) model", () => {
    const model = Waifu2xModel.fromJSON(vggLikeModel(1));
    const image = { rgb: randomImage(3, 12, 14, 4) };
    const result = reconstructScale(model, 2, image.rgb, { engine: "fast", blockSize: 24 });
    expect(result.width).toBe(28);
    expect(result.height).toBe(24);
  });

  it("preserves and upscales alpha through the scale pipeline", () => {
    const model = Waifu2xModel.fromJSON(upconvLikeModel(3));
    const rgb = randomImage(3, 16, 16, 1);
    const alpha = randomImage(1, 16, 16, 2);
    const result = scaleImage(model, 2, { rgb, alpha }, { engine: "fast", blockSize: 24, alphaScale: "lanczos" });
    expect(result.alpha?.width).toBe(32);
    expect(result.alpha?.height).toBe(32);
  });
});
