import { describe, expect, it } from "vitest";
import { createTensor } from "../src/core/tensor.js";
import { brainConvForward } from "../src/engine/brainConv.js";
import { buildNetwork, convNetworkJSON, IDENTITY_ALPHA } from "../src/brain/network.js";
import { spatialFullConvolution, deconvOutputSize } from "../src/engine/deconv.js";
import { rgb2yuv, yuv2rgb } from "../src/image/color.js";
import { resizeNearest } from "../src/image/resize.js";
import { padEdge, crop } from "../src/image/pad.js";
import type { ConvLayer, DeconvLayer } from "../src/model/types.js";
import { randomImage, maxAbsDiff } from "./helpers.js";

describe("brain conv", () => {
  it("matches a hand-computed 3x3 valid convolution", () => {
    const input = createTensor(1, 3, 3);
    input.data.set([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const layer: ConvLayer = {
      kind: "conv",
      inputPlanes: 1,
      outputPlanes: 1,
      kernelWidth: 3,
      kernelHeight: 3,
      strideX: 1,
      strideY: 1,
      padX: 0,
      padY: 0,
    };
    const net = buildNetwork(convNetworkJSON([[0, 0, 0, 0, 1, 0, 0, 0, 0]], [10], IDENTITY_ALPHA));
    const out = brainConvForward(input, layer, net);
    expect(out.width).toBe(1);
    expect(out.height).toBe(1);
    expect(out.data[0]).toBeCloseTo(15, 5);
  });

  it("sums two input planes", () => {
    const input = createTensor(2, 1, 1);
    input.data.set([3, 4]);
    const layer: ConvLayer = {
      kind: "conv",
      inputPlanes: 2,
      outputPlanes: 1,
      kernelWidth: 1,
      kernelHeight: 1,
      strideX: 1,
      strideY: 1,
      padX: 0,
      padY: 0,
    };
    const net = buildNetwork(convNetworkJSON([[2, 5]], [1], IDENTITY_ALPHA));
    const out = brainConvForward(input, layer, net);
    expect(out.data[0]).toBeCloseTo(2 * 3 + 5 * 4 + 1, 5);
  });
});

describe("deconv", () => {
  it("computes the upconv output size", () => {
    expect(deconvOutputSize(116, 4, 2, 3, 0)).toBe(228);
    expect(deconvOutputSize(10, 4, 2, 3, 0)).toBe(16);
  });

  it("upsamples by a factor of two in each dimension", () => {
    const input = randomImage(2, 8, 8, 3);
    const layer: DeconvLayer = {
      kind: "deconv",
      inputPlanes: 2,
      outputPlanes: 3,
      kernelWidth: 4,
      kernelHeight: 4,
      strideX: 2,
      strideY: 2,
      padX: 3,
      padY: 3,
      adjX: 0,
      adjY: 0,
      weights: new Float32Array(2 * 3 * 4 * 4).fill(0.01),
      bias: Float32Array.from([0, 0, 0]),
    };
    const out = spatialFullConvolution(input, layer);
    expect(out.channels).toBe(3);
    expect(out.width).toBe(2 * 8 - 4);
    expect(out.height).toBe(2 * 8 - 4);
  });
});

describe("color", () => {
  it("round-trips rgb -> yuv -> rgb", () => {
    const rgb = randomImage(3, 5, 6, 11);
    const back = yuv2rgb(rgb2yuv(rgb));
    expect(maxAbsDiff(rgb, back)).toBeLessThan(1e-4);
  });
});

describe("resize", () => {
  it("duplicates pixels on a 2x nearest upscale", () => {
    const input = createTensor(1, 2, 2);
    input.data.set([1, 2, 3, 4]);
    const out = resizeNearest(input, 4, 4);
    expect(Array.from(out.data)).toEqual([1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4]);
  });
});

describe("pad/crop", () => {
  it("replicates edges and crops back to the original", () => {
    const input = randomImage(1, 4, 4, 5);
    const padded = padEdge(input, 2, 2, 2, 2);
    expect(padded.width).toBe(8);
    expect(padded.height).toBe(8);
    const cropped = crop(padded, 2, 2, 6, 6);
    expect(maxAbsDiff(cropped, input)).toBe(0);
  });
});
