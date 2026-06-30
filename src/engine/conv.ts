export function convOutputSize(inputSize: number, kernel: number, stride: number, pad: number): number {
  return Math.floor((inputSize + 2 * pad - kernel) / stride) + 1;
}
