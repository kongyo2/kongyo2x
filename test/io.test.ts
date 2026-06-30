import { describe, expect, it } from "vitest";
import { encodePng, decodeImage } from "../src/image/io.js";
import { randomImage, maxAbsDiff } from "./helpers.js";

describe("png io", () => {
  it("round-trips an RGB image within 8-bit quantization error", () => {
    const rgb = randomImage(3, 9, 7, 99);
    const decoded = decodeImage(encodePng(rgb));
    expect(decoded.rgb.width).toBe(7);
    expect(decoded.rgb.height).toBe(9);
    expect(decoded.alpha).toBeUndefined();
    expect(maxAbsDiff(rgb, decoded.rgb)).toBeLessThanOrEqual(1 / 255 + 1e-6);
  });

  it("preserves an alpha channel", () => {
    const rgb = randomImage(3, 5, 5, 1);
    const alpha = randomImage(1, 5, 5, 2);
    alpha.data[0] = 0;
    const decoded = decodeImage(encodePng(rgb, alpha));
    expect(decoded.alpha).toBeDefined();
    expect((decoded.alpha?.data[0] as number) <= 1 / 255).toBe(true);
  });
});
