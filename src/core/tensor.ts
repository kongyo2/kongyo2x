export interface Tensor {
  readonly channels: number;
  readonly height: number;
  readonly width: number;
  readonly data: Float32Array;
}

export function createTensor(channels: number, height: number, width: number): Tensor {
  return { channels, height, width, data: new Float32Array(channels * height * width) };
}

export function fromData(channels: number, height: number, width: number, data: Float32Array): Tensor {
  if (data.length !== channels * height * width) {
    throw new Error(`tensor data length ${data.length} does not match shape ${channels}x${height}x${width}`);
  }
  return { channels, height, width, data };
}

export function planeOffset(t: Tensor, channel: number): number {
  return channel * t.height * t.width;
}

export function at(t: Tensor, channel: number, y: number, x: number): number {
  return t.data[(channel * t.height + y) * t.width + x] as number;
}

export function setAt(t: Tensor, channel: number, y: number, x: number, value: number): void {
  t.data[(channel * t.height + y) * t.width + x] = value;
}

export function cloneTensor(t: Tensor): Tensor {
  return { channels: t.channels, height: t.height, width: t.width, data: Float32Array.from(t.data) };
}

export function getChannel(t: Tensor, channel: number): Tensor {
  const size = t.height * t.width;
  const start = channel * size;
  return { channels: 1, height: t.height, width: t.width, data: t.data.subarray(start, start + size) };
}

export function stackChannels(channels: Tensor[]): Tensor {
  if (channels.length === 0) {
    throw new Error("stackChannels requires at least one channel");
  }
  const first = channels[0] as Tensor;
  const { height, width } = first;
  const size = height * width;
  const out = new Float32Array(channels.length * size);
  for (let c = 0; c < channels.length; c++) {
    const plane = channels[c] as Tensor;
    if (plane.height !== height || plane.width !== width) {
      throw new Error("stackChannels requires identical plane dimensions");
    }
    out.set(plane.data.subarray(0, size), c * size);
  }
  return { channels: channels.length, height, width, data: out };
}

export function clamp01(t: Tensor): Tensor {
  const data = t.data;
  for (let i = 0; i < data.length; i++) {
    const v = data[i] as number;
    data[i] = v < 0 ? 0 : v > 1 ? 1 : v;
  }
  return t;
}
