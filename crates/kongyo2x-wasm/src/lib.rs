//! WebAssembly numeric kernels for kongyo2x.
//!
//! Every kernel mirrors its TypeScript reference exactly: values are stored as
//! `f32` but every multiply-accumulate runs through an `f64` accumulator, which
//! is what JavaScript does implicitly when it reads a `Float32Array` element
//! into a `number`. That keeps the Wasm and pure-TS backends bit-identical.

// The kernels use a flat C ABI (many scalar shape args by design) and a manual
// `v<0?0:v>1?1:v` clamp chosen to match JavaScript's clamp semantics exactly.
#![allow(clippy::too_many_arguments, clippy::manual_clamp)]

use std::alloc::{alloc, dealloc, Layout};
use std::slice;

const ALIGN: usize = 16;

#[no_mangle]
pub extern "C" fn kw_alloc(size: usize) -> *mut u8 {
    if size == 0 {
        return ALIGN as *mut u8;
    }
    let layout = Layout::from_size_align(size, ALIGN).unwrap();
    // SAFETY: size is non-zero and the alignment is a valid power of two.
    let ptr = unsafe { alloc(layout) };
    if ptr.is_null() {
        std::process::abort();
    }
    ptr
}

/// # Safety
/// `(ptr, size)` must be a live allocation returned by `kw_alloc`.
#[no_mangle]
pub unsafe extern "C" fn kw_dealloc(ptr: *mut u8, size: usize) {
    if size == 0 || ptr.is_null() {
        return;
    }
    let layout = Layout::from_size_align(size, ALIGN).unwrap();
    // SAFETY: (ptr, size) was produced by kw_alloc with the same alignment.
    dealloc(ptr, layout);
}

#[inline]
unsafe fn fin<'a>(ptr: *const f32, len: usize) -> &'a [f32] {
    slice::from_raw_parts(ptr, len)
}

#[inline]
unsafe fn fout<'a>(ptr: *mut f32, len: usize) -> &'a mut [f32] {
    slice::from_raw_parts_mut(ptr, len)
}

#[inline]
unsafe fn iin<'a>(ptr: *const i32, len: usize) -> &'a [i32] {
    slice::from_raw_parts(ptr, len)
}

/// Inference convolution with a fused leaky-ReLU, mirroring `cpuConvForward`.
///
/// # Safety
/// All pointers must reference `f32` regions of the documented lengths.
#[no_mangle]
pub unsafe extern "C" fn conv_forward(
    in_ptr: *const f32,
    in_h: u32,
    in_w: u32,
    w_ptr: *const f32,
    b_ptr: *const f32,
    out_ptr: *mut f32,
    input_planes: u32,
    output_planes: u32,
    kw: u32,
    kh: u32,
    stride_x: u32,
    stride_y: u32,
    pad_x: u32,
    pad_y: u32,
    alpha: f64,
) {
    let in_h = in_h as isize;
    let in_w = in_w as isize;
    let ip = input_planes as isize;
    let op = output_planes as isize;
    let kw = kw as isize;
    let kh = kh as isize;
    let sx = stride_x as isize;
    let sy = stride_y as isize;
    let px = pad_x as isize;
    let py = pad_y as isize;
    let out_h = (in_h + 2 * py - kh) / sy + 1;
    let out_w = (in_w + 2 * px - kw) / sx + 1;

    let in_data = fin(in_ptr, (ip * in_h * in_w) as usize);
    let weights = fin(w_ptr, (op * ip * kh * kw) as usize);
    let bias = fin(b_ptr, op as usize);
    let out = fout(out_ptr, (op * out_h * out_w) as usize);

    let weight_stride = ip * kh * kw;
    let plane_size = out_h * out_w;

    // Valid convolution (stride 1, no padding) — the shape every kongyo2x conv
    // layer uses. Every tap is in range, so the per-tap bounds check is dropped.
    if px == 0 && py == 0 && sx == 1 && sy == 1 {
        for o in 0..op {
            let bias_value = bias[o as usize] as f64;
            let w_base = o * weight_stride;
            let out_plane = o * plane_size;
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let mut sum = bias_value;
                    let mut w = w_base;
                    for i in 0..ip {
                        let plane_base = i * in_h * in_w;
                        for ky in 0..kh {
                            let row_base = plane_base + (oy + ky) * in_w + ox;
                            for kx in 0..kw {
                                sum += (*in_data.get_unchecked((row_base + kx) as usize) as f64)
                                    * (*weights.get_unchecked(w as usize) as f64);
                                w += 1;
                            }
                        }
                    }
                    let scaled = alpha * sum;
                    let val = if sum >= scaled { sum } else { scaled };
                    *out.get_unchecked_mut((out_plane + oy * out_w + ox) as usize) = val as f32;
                }
            }
        }
        return;
    }

    for o in 0..op {
        let bias_value = bias[o as usize] as f64;
        let w_base = o * weight_stride;
        let out_plane = o * plane_size;
        for oy in 0..out_h {
            let base_y = oy * sy - py;
            for ox in 0..out_w {
                let base_x = ox * sx - px;
                let mut sum = bias_value;
                let mut w = w_base;
                for i in 0..ip {
                    let plane_base = i * in_h * in_w;
                    for ky in 0..kh {
                        let iy = base_y + ky;
                        let row_in_range = iy >= 0 && iy < in_h;
                        let row_base = plane_base + iy * in_w;
                        for kx in 0..kw {
                            let ix = base_x + kx;
                            if row_in_range && ix >= 0 && ix < in_w {
                                sum += (*in_data.get_unchecked((row_base + ix) as usize) as f64)
                                    * (*weights.get_unchecked(w as usize) as f64);
                            }
                            w += 1;
                        }
                    }
                }
                let scaled = alpha * sum;
                let val = if sum >= scaled { sum } else { scaled };
                *out.get_unchecked_mut((out_plane + oy * out_w + ox) as usize) = val as f32;
            }
        }
    }
}

/// Transposed convolution (upsampling), mirroring `spatialFullConvolution`.
///
/// # Safety
/// All pointers must reference `f32` regions of the documented lengths.
#[no_mangle]
pub unsafe extern "C" fn deconv_forward(
    in_ptr: *const f32,
    in_h: u32,
    in_w: u32,
    w_ptr: *const f32,
    b_ptr: *const f32,
    out_ptr: *mut f32,
    input_planes: u32,
    output_planes: u32,
    kw: u32,
    kh: u32,
    stride_x: u32,
    stride_y: u32,
    pad_x: u32,
    pad_y: u32,
    adj_x: u32,
    adj_y: u32,
) {
    let in_h = in_h as isize;
    let in_w = in_w as isize;
    let ip = input_planes as isize;
    let op = output_planes as isize;
    let kw = kw as isize;
    let kh = kh as isize;
    let sx = stride_x as isize;
    let sy = stride_y as isize;
    let px = pad_x as isize;
    let py = pad_y as isize;
    let out_h = (in_h - 1) * sy - 2 * py + kh + adj_y as isize;
    let out_w = (in_w - 1) * sx - 2 * px + kw + adj_x as isize;

    let in_data = fin(in_ptr, (ip * in_h * in_w) as usize);
    let weights = fin(w_ptr, (ip * op * kh * kw) as usize);
    let bias = fin(b_ptr, op as usize);
    let out = fout(out_ptr, (op * out_h * out_w) as usize);

    let plane_size = out_h * out_w;
    for o in 0..op {
        let bias_value = *bias.get_unchecked(o as usize);
        let base = (o * plane_size) as usize;
        for slot in out.get_unchecked_mut(base..base + plane_size as usize) {
            *slot = bias_value;
        }
    }

    for i in 0..ip {
        let in_plane_base = i * in_h * in_w;
        let weight_in_base = i * op * kh * kw;
        for iy in 0..in_h {
            for ix in 0..in_w {
                let v = *in_data.get_unchecked((in_plane_base + iy * in_w + ix) as usize);
                if v == 0.0 {
                    continue;
                }
                let v = v as f64;
                let oy_base = iy * sy - py;
                let ox_base = ix * sx - px;
                for o in 0..op {
                    let weight_base = weight_in_base + o * kh * kw;
                    let out_plane_base = o * plane_size;
                    for ky in 0..kh {
                        let oy = oy_base + ky;
                        if oy < 0 || oy >= out_h {
                            continue;
                        }
                        let out_row_base = out_plane_base + oy * out_w;
                        let weight_row_base = weight_base + ky * kw;
                        for kx in 0..kw {
                            let ox = ox_base + kx;
                            if ox >= 0 && ox < out_w {
                                let pos = (out_row_base + ox) as usize;
                                let acc = *out.get_unchecked(pos) as f64
                                    + (*weights.get_unchecked((weight_row_base + kx) as usize)
                                        as f64)
                                        * v;
                                *out.get_unchecked_mut(pos) = acc as f32;
                            }
                        }
                    }
                }
            }
        }
    }
}

#[inline]
unsafe fn im2col(
    in_data: &[f32],
    in_h: isize,
    in_w: isize,
    cin: isize,
    kh: isize,
    kw: isize,
    out_h: isize,
    out_w: isize,
    k: isize,
    col: &mut [f32],
) {
    let plane_in = in_h * in_w;
    for oy in 0..out_h {
        for ox in 0..out_w {
            let mut dst = (oy * out_w + ox) * k;
            for i in 0..cin {
                let in_plane = i * plane_in;
                for ky in 0..kh {
                    let in_row = in_plane + (oy + ky) * in_w + ox;
                    for kx in 0..kw {
                        *col.get_unchecked_mut(dst as usize) =
                            *in_data.get_unchecked((in_row + kx) as usize);
                        dst += 1;
                    }
                }
            }
        }
    }
}

/// Training-time valid convolution (im2col + GEMM), mirroring `convForward`.
///
/// # Safety
/// All pointers must reference `f32` regions of the documented lengths.
#[no_mangle]
pub unsafe extern "C" fn conv_forward_train(
    in_ptr: *const f32,
    in_h: u32,
    in_w: u32,
    w_ptr: *const f32,
    b_ptr: *const f32,
    out_ptr: *mut f32,
    cin: u32,
    cout: u32,
    kh: u32,
    kw: u32,
) {
    let in_h = in_h as isize;
    let in_w = in_w as isize;
    let cin = cin as isize;
    let cout = cout as isize;
    let kh = kh as isize;
    let kw = kw as isize;
    let out_h = in_h - kh + 1;
    let out_w = in_w - kw + 1;
    let k = cin * kh * kw;
    let p = out_h * out_w;

    let in_data = fin(in_ptr, (cin * in_h * in_w) as usize);
    let weights = fin(w_ptr, (cout * k) as usize);
    let bias = fin(b_ptr, cout as usize);
    let out = fout(out_ptr, (cout * p) as usize);

    let mut col = vec![0.0f32; (p * k) as usize];
    im2col(in_data, in_h, in_w, cin, kh, kw, out_h, out_w, k, &mut col);

    for o in 0..cout {
        let w_base = o * k;
        let out_base = o * p;
        let bias_value = *bias.get_unchecked(o as usize) as f64;
        for pi in 0..p {
            let col_base = pi * k;
            let mut acc = bias_value;
            for ki in 0..k {
                acc += (*weights.get_unchecked((w_base + ki) as usize) as f64)
                    * (*col.get_unchecked((col_base + ki) as usize) as f64);
            }
            *out.get_unchecked_mut((out_base + pi) as usize) = acc as f32;
        }
    }
}

/// Training-time convolution backward pass, mirroring `convBackward`.
///
/// `grad_w_ptr`/`grad_b_ptr` are read-modify-write accumulators. When
/// `compute_grad_input` is non-zero, `din_ptr` receives the input gradient
/// (its region is zeroed first).
///
/// # Safety
/// All pointers must reference `f32` regions of the documented lengths.
#[no_mangle]
pub unsafe extern "C" fn conv_backward_train(
    in_ptr: *const f32,
    in_h: u32,
    in_w: u32,
    w_ptr: *const f32,
    dpre_ptr: *const f32,
    out_h_in: u32,
    out_w_in: u32,
    grad_w_ptr: *mut f32,
    grad_b_ptr: *mut f32,
    din_ptr: *mut f32,
    cin: u32,
    cout: u32,
    kh: u32,
    kw: u32,
    compute_grad_input: u32,
) {
    let in_h = in_h as isize;
    let in_w = in_w as isize;
    let cin = cin as isize;
    let cout = cout as isize;
    let kh = kh as isize;
    let kw = kw as isize;
    let out_h = out_h_in as isize;
    let out_w = out_w_in as isize;
    let k = cin * kh * kw;
    let p = out_h * out_w;

    let in_data = fin(in_ptr, (cin * in_h * in_w) as usize);
    let weights = fin(w_ptr, (cout * k) as usize);
    let dpre = fin(dpre_ptr, (cout * p) as usize);
    let grad_w = fout(grad_w_ptr, (cout * k) as usize);
    let grad_b = fout(grad_b_ptr, cout as usize);

    let mut col = vec![0.0f32; (p * k) as usize];
    im2col(in_data, in_h, in_w, cin, kh, kw, out_h, out_w, k, &mut col);

    for o in 0..cout {
        let w_base = o * k;
        let out_base = o * p;
        let mut bias_acc = 0.0f64;
        for pi in 0..p {
            let g = *dpre.get_unchecked((out_base + pi) as usize) as f64;
            bias_acc += g;
            if g == 0.0 {
                continue;
            }
            let col_base = pi * k;
            for ki in 0..k {
                let idx = (w_base + ki) as usize;
                let acc = *grad_w.get_unchecked(idx) as f64
                    + g * (*col.get_unchecked((col_base + ki) as usize) as f64);
                *grad_w.get_unchecked_mut(idx) = acc as f32;
            }
        }
        let gb = *grad_b.get_unchecked(o as usize) as f64 + bias_acc;
        *grad_b.get_unchecked_mut(o as usize) = gb as f32;
    }

    if compute_grad_input == 0 {
        return;
    }

    let din = fout(din_ptr, (cin * in_h * in_w) as usize);
    for slot in din.iter_mut() {
        *slot = 0.0;
    }

    let mut dcol = vec![0.0f32; (p * k) as usize];
    for o in 0..cout {
        let w_base = o * k;
        let out_base = o * p;
        for pi in 0..p {
            let g = *dpre.get_unchecked((out_base + pi) as usize) as f64;
            if g == 0.0 {
                continue;
            }
            let col_base = pi * k;
            for ki in 0..k {
                let idx = (col_base + ki) as usize;
                let acc = *dcol.get_unchecked(idx) as f64
                    + g * (*weights.get_unchecked((w_base + ki) as usize) as f64);
                *dcol.get_unchecked_mut(idx) = acc as f32;
            }
        }
    }

    let plane_in = in_h * in_w;
    for oy in 0..out_h {
        for ox in 0..out_w {
            let col_base = (oy * out_w + ox) * k;
            for i in 0..cin {
                let in_plane = i * plane_in;
                let k_base = col_base + i * kh * kw;
                for ky in 0..kh {
                    let in_row = in_plane + (oy + ky) * in_w + ox;
                    let k_row = k_base + ky * kw;
                    for kx in 0..kw {
                        let didx = (in_row + kx) as usize;
                        let acc = *din.get_unchecked(didx) as f64
                            + *dcol.get_unchecked((k_row + kx) as usize) as f64;
                        *din.get_unchecked_mut(didx) = acc as f32;
                    }
                }
            }
        }
    }
}

/// Separable Lanczos resample, mirroring `resizeLanczos`. Taps (indices and
/// normalized weights) are precomputed by the caller so the transcendental
/// `sin` math stays identical to the TypeScript reference.
///
/// # Safety
/// All pointers must reference regions of the documented lengths and types.
#[no_mangle]
pub unsafe extern "C" fn resize_lanczos(
    in_ptr: *const f32,
    channels: u32,
    in_h: u32,
    in_w: u32,
    x_idx_ptr: *const i32,
    x_w_ptr: *const f32,
    x_taps: u32,
    out_w_in: u32,
    y_idx_ptr: *const i32,
    y_w_ptr: *const f32,
    y_taps: u32,
    out_h_in: u32,
    out_ptr: *mut f32,
) {
    let channels = channels as isize;
    let in_h = in_h as isize;
    let in_w = in_w as isize;
    let out_w = out_w_in as isize;
    let out_h = out_h_in as isize;
    let x_taps = x_taps as isize;
    let y_taps = y_taps as isize;

    let in_data = fin(in_ptr, (channels * in_h * in_w) as usize);
    let x_idx = iin(x_idx_ptr, (out_w * x_taps) as usize);
    let x_w = fin(x_w_ptr, (out_w * x_taps) as usize);
    let y_idx = iin(y_idx_ptr, (out_h * y_taps) as usize);
    let y_w = fin(y_w_ptr, (out_h * y_taps) as usize);
    let out = fout(out_ptr, (channels * out_h * out_w) as usize);

    let mut horizontal = vec![0.0f32; (channels * in_h * out_w) as usize];
    for c in 0..channels {
        let src_plane = c * in_h * in_w;
        let dst_plane = c * in_h * out_w;
        for y in 0..in_h {
            let src_row = src_plane + y * in_w;
            let dst_row = dst_plane + y * out_w;
            for ox in 0..out_w {
                let base = ox * x_taps;
                let mut acc = 0.0f64;
                for t in 0..x_taps {
                    acc += (*x_w.get_unchecked((base + t) as usize) as f64)
                        * (*in_data.get_unchecked(
                            (src_row + *x_idx.get_unchecked((base + t) as usize) as isize) as usize,
                        ) as f64);
                }
                *horizontal.get_unchecked_mut((dst_row + ox) as usize) = acc as f32;
            }
        }
    }

    for c in 0..channels {
        let src_plane = c * in_h * out_w;
        let dst_plane = c * out_h * out_w;
        for oy in 0..out_h {
            let base = oy * y_taps;
            let dst_row = dst_plane + oy * out_w;
            for ox in 0..out_w {
                let mut acc = 0.0f64;
                for t in 0..y_taps {
                    let sy = *y_idx.get_unchecked((base + t) as usize) as isize;
                    acc += (*y_w.get_unchecked((base + t) as usize) as f64)
                        * (*horizontal.get_unchecked((src_plane + sy * out_w + ox) as usize)
                            as f64);
                }
                *out.get_unchecked_mut((dst_row + ox) as usize) = acc as f32;
            }
        }
    }
}

#[inline]
fn box3x3_sum(plane: &[f32], height: isize, width: isize) -> Vec<f32> {
    let mut out = vec![0.0f32; (height * width) as usize];
    for y in 0..height {
        let y0 = if y > 0 { y - 1 } else { 0 };
        let y1 = if y < height - 1 { y + 1 } else { height - 1 };
        for x in 0..width {
            let x0 = if x > 0 { x - 1 } else { 0 };
            let x1 = if x < width - 1 { x + 1 } else { width - 1 };
            let mut sum = 0.0f64;
            for yy in y0..=y1 {
                let row_base = yy * width;
                for xx in x0..=x1 {
                    sum += plane[(row_base + xx) as usize] as f64;
                }
            }
            out[(y * width + x) as usize] = sum as f32;
        }
    }
    out
}

/// Alpha-aware edge extension, mirroring `makeBorder`. Writes the bordered RGB
/// (clamped to `[0,1]`) into `out_ptr`.
///
/// # Safety
/// All pointers must reference `f32` regions of the documented lengths.
#[no_mangle]
pub unsafe extern "C" fn make_border(
    rgb_ptr: *const f32,
    alpha_ptr: *const f32,
    out_ptr: *mut f32,
    height: u32,
    width: u32,
    offset: u32,
) {
    let height = height as isize;
    let width = width as isize;
    let size = (height * width) as usize;
    let eps = 1e-7f64;

    let rgb = fin(rgb_ptr, size * 3);
    let alpha = fin(alpha_ptr, size);
    let out = fout(out_ptr, size * 3);
    out.copy_from_slice(rgb);

    let mut mask = vec![0.0f32; size];
    for p in 0..size {
        mask[p] = if alpha[p] > 0.0 { 1.0 } else { 0.0 };
    }
    for p in 0..size {
        if mask[p] == 0.0 {
            out[p] = 0.0;
            out[size + p] = 0.0;
            out[2 * size + p] = 0.0;
        }
    }

    for _ in 0..offset {
        let mask_weight = box3x3_sum(&mask, height, width);
        for ch in 0..3usize {
            let channel_base = ch * size;
            let blurred = box3x3_sum(&out[channel_base..channel_base + size], height, width);
            for p in 0..size {
                if mask[p] == 0.0 {
                    let value = (blurred[p] as f64) / (mask_weight[p] as f64 + eps);
                    out[channel_base + p] = value as f32;
                }
            }
        }
        for p in 0..size {
            mask[p] = if mask_weight[p] > 0.0 { 1.0 } else { 0.0 };
        }
    }

    for slot in out.iter_mut() {
        let v = *slot;
        *slot = if v < 0.0 {
            0.0
        } else if v > 1.0 {
            1.0
        } else {
            v
        };
    }
}
