//! Whole-frame matting: every frame leaves as a grayscale alpha of its
//! foreground - 255 where the subject owns the pixel, 0 where the background
//! does, and the values between are the blend.
//!
//! The graph is longpipe's small matting net, run through `wasi:nn` at its
//! fixed 320x192 input. The net has no classes: its one output is the
//! foreground alpha, so there is nothing to narrow and no parameters to
//! take. The frame is stretch-resized to the model's size - no letterbox, no
//! crop, the way the runtime the weights shipped in feeds it - as RGB
//! rescaled to 0..1 with no further normalization, and the alpha that comes
//! back is resized bilinearly onto the frame's own geometry. The module
//! never opens a file - the host binds the graph to a name with
//! `-nn matte=<path>` and this module asks for that name and nothing else.
//!
//! The matte keeps the frame's geometry and pixel format, so it feeds
//! straight into whatever reads a mask beside the picture: in yuv420p the
//! alpha is the luma with neutral chroma, in rgba the same value in red,
//! green and blue, opaque.

// `generate_all`: the world's interfaces come from two other packages -
// ffrwd:av and wasi:nn - and without it bindgen expects them to have been
// generated somewhere else.
wit_bindgen::generate!({
    path: ["wit", "wit-world"],
    // Fully qualified: three packages are in scope, and each has worlds.
    world: "ffrwd:longpipe/matte",
    generate_all,
});

use std::cell::{Cell, RefCell};

use exports::ffrwd::av::window_filter::{
    Format, FramePayload, Guest, InWindow, Meta, OutFrame, Processed, StreamInfo, WindowMeta,
};
use wasi::nn::graph::{load_by_name, Graph};
use wasi::nn::inference::GraphExecutionContext;
use wasi::nn::tensor::{Tensor, TensorType};

/// The name the host binds the graph to. `-nn matte=<path>`.
const MODEL: &str = "matte";

/// The size the graph is exported at, fixed: the TF-style pads inside it
/// trace to constants, so the export carries exactly one geometry.
const MODEL_W: usize = 320;
const MODEL_H: usize = 192;

/// What the export calls its input tensor.
const INPUT_NAME: &str = "rgb";

/// The host accepts a position where it accepts a name, which is what an
/// export that named its input something else is reached by.
const INPUT_INDEX: &str = "0";

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;

/// The pixel format an instance was opened for, fixed for its life.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PixFmt {
    Yuv420p,
    Rgba,
}

impl PixFmt {
    /// The format the host named, or an error naming what it was.
    fn parse(named: &str) -> Result<PixFmt, String> {
        match named {
            "yuv420p" => Ok(PixFmt::Yuv420p),
            "rgba" => Ok(PixFmt::Rgba),
            other => Err(format!("matte does not accept pixel format {other}")),
        }
    }
}

/// What `init` settled, plus the graph it loaded.
struct Opened {
    width: usize,
    height: usize,
    pix_fmt: PixFmt,
    /// What the graph calls its input, settled by the first call that works.
    input_name: Cell<&'static str>,
    /// Held for the life of the instance: building it once is what keeps a
    /// provider's kernels from being chosen again per frame.
    context: GraphExecutionContext,
    /// Kept alive because the context is only valid while its graph is.
    _graph: Graph,
}

thread_local! {
    static OPENED: RefCell<Option<Opened>> = const { RefCell::new(None) };
}

/// This module takes no parameters: the net has no classes and no threshold.
fn validate_params(params: &str) -> Result<(), String> {
    match params.trim() {
        "" | "{}" => Ok(()),
        other => Err(format!("matte takes no params, got: {other}")),
    }
}

/// The spec's spelling of an error code, so a message says what actually
/// went wrong rather than how this module happens to format things.
fn failed(what: &str, error: &wasi::nn::errors::Error) -> String {
    use wasi::nn::errors::ErrorCode;
    let code = match error.code() {
        ErrorCode::InvalidArgument => "invalid-argument",
        ErrorCode::InvalidEncoding => "invalid-encoding",
        ErrorCode::Timeout => "timeout",
        ErrorCode::RuntimeError => "runtime-error",
        ErrorCode::UnsupportedOperation => "unsupported-operation",
        ErrorCode::TooLarge => "too-large",
        ErrorCode::NotFound => "not-found",
        ErrorCode::Security => "security",
        ErrorCode::Unknown => "unknown",
    };
    format!("matte: {what}: {code} ({})", error.data())
}

/// Where each of `count` output steps reads from along a source `extent`
/// samples long: the two samples it falls between, and how far along it
/// sits. The mapping is the bilinear `(step + 0.5) * extent / count - 0.5`,
/// which is what the model's own runtime and its exported upsamples use.
struct Taps {
    low: Vec<usize>,
    high: Vec<usize>,
    fraction: Vec<f32>,
}

impl Taps {
    fn build(count: usize, extent: usize) -> Taps {
        let mut map = Taps {
            low: Vec::with_capacity(count),
            high: Vec::with_capacity(count),
            fraction: Vec::with_capacity(count),
        };
        let per_step = extent as f32 / count as f32;
        for step in 0..count {
            let f = ((step as f32 + 0.5) * per_step - 0.5).clamp(0.0, (extent - 1) as f32);
            let base = f.floor() as usize;
            map.low.push(base);
            map.high.push((base + 1).min(extent - 1));
            map.fraction.push(f - base as f32);
        }
        map
    }
}

/// One frame row as red, green and blue, a channel at a time so each is
/// contiguous.
fn row_to_rgb(frame: &[u8], pix_fmt: PixFmt, width: usize, height: usize, y: usize, out: &mut [f32]) {
    let (red, rest) = out.split_at_mut(width);
    let (green, blue) = rest.split_at_mut(width);
    match pix_fmt {
        PixFmt::Rgba => {
            for (x, pixel) in frame[y * width * 4..(y + 1) * width * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .enumerate()
            {
                red[x] = f32::from(pixel[0]);
                green[x] = f32::from(pixel[1]);
                blue[x] = f32::from(pixel[2]);
            }
        }
        PixFmt::Yuv420p => {
            let pixels = width * height;
            let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
            let chroma = cw * ch;
            let luma = &frame[y * width..(y + 1) * width];
            let crow = (y / 2).min(ch - 1) * cw;
            for x in 0..width {
                let l = f32::from(luma[x]);
                let ci = crow + (x / 2).min(cw - 1);
                let u = f32::from(frame[pixels + ci]) - 128.0;
                let v = f32::from(frame[pixels + chroma + ci]) - 128.0;
                // The usual BT.601 inverse, in full range: the frames a module
                // is handed are what the host decoded, not studio-swing video.
                red[x] = l + 1.402 * v;
                green[x] = l - 0.344_136 * u - 0.714_136 * v;
                blue[x] = l + 1.772 * u;
            }
        }
    }
}

/// One frame row resized to the model's columns, channel by channel.
fn resize_rgb_row(rgb: &[f32], columns: &Taps, width: usize, out: &mut [f32]) {
    for channel in 0..3 {
        let source = &rgb[channel * width..(channel + 1) * width];
        let target = &mut out[channel * MODEL_W..(channel + 1) * MODEL_W];
        for (((sample, low), high), fraction) in target
            .iter_mut()
            .zip(&columns.low)
            .zip(&columns.high)
            .zip(&columns.fraction)
        {
            let (a, b) = (source[*low], source[*high]);
            *sample = a + (b - a) * fraction;
        }
    }
}

/// The frame stretched to the model's 320x192 - each axis by its own ratio,
/// no letterbox, the contract the weights shipped with - and laid out as the
/// planar fp32 tensor the graph expects: red, green and blue in turn, each
/// rescaled to 0..1 and nothing else.
fn to_input(frame: &[u8], pix_fmt: PixFmt, width: usize, height: usize) -> Vec<u8> {
    let plane = MODEL_W * MODEL_H;
    let mut planes = vec![0f32; plane * 3];

    let columns = Taps::build(MODEL_W, width);
    let rows = Taps::build(MODEL_H, height);
    let mut rgb = vec![0f32; width * 3];
    let mut top = vec![0f32; MODEL_W * 3];
    let mut bottom = vec![0f32; MODEL_W * 3];

    for my in 0..MODEL_H {
        row_to_rgb(frame, pix_fmt, width, height, rows.low[my], &mut rgb);
        resize_rgb_row(&rgb, &columns, width, &mut top);
        if rows.high[my] != rows.low[my] {
            row_to_rgb(frame, pix_fmt, width, height, rows.high[my], &mut rgb);
            resize_rgb_row(&rgb, &columns, width, &mut bottom);
        } else {
            bottom.copy_from_slice(&top);
        }
        let ty = rows.fraction[my];

        for channel in 0..3 {
            let a = &top[channel * MODEL_W..(channel + 1) * MODEL_W];
            let b = &bottom[channel * MODEL_W..(channel + 1) * MODEL_W];
            let target = &mut planes[channel * plane + my * MODEL_W..][..MODEL_W];
            for ((sample, ta), tb) in target.iter_mut().zip(a).zip(b) {
                *sample = (ta + (tb - ta) * ty).clamp(0.0, 255.0) / 255.0;
            }
        }
    }

    let mut bytes = vec![0u8; planes.len() * 4];
    let (words, _) = bytes.as_chunks_mut::<4>();
    for (word, value) in words.iter_mut().zip(&planes) {
        *word = value.to_le_bytes();
    }
    bytes
}

/// A tensor's floats, out of the little-endian bytes it arrived as.
fn le_f32s(data: &[u8]) -> Vec<f32> {
    let (whole, _) = data.as_chunks::<4>();
    whole.iter().copied().map(f32::from_le_bytes).collect()
}

/// Which returned tensor is the alpha, by shape: the one whose trailing two
/// dimensions are the model's own 192x320 plane and whose leading ones are
/// all 1. Names are not read, so an export that spells its output
/// differently still resolves.
fn output(shapes: &[Vec<u32>]) -> Result<usize, String> {
    let alpha = shapes.iter().position(|dimensions| {
        matches!(dimensions.as_slice(),
            [rest @ .., height, width]
                if *height as usize == MODEL_H
                    && *width as usize == MODEL_W
                    && rest.iter().all(|d| *d == 1))
    });
    alpha.ok_or_else(|| {
        format!(
            "matte: the graph returned {shapes:?}, and this module wants the \
             [1, 1, {MODEL_H}, {MODEL_W}] alpha of longpipe's small export"
        )
    })
}

/// The model's alpha brought onto the frame's own geometry, bilinearly, as
/// 8-bit gray.
fn upscale_alpha(alpha: &[f32], width: usize, height: usize) -> Vec<u8> {
    let columns = Taps::build(width, MODEL_W);
    let rows = Taps::build(height, MODEL_H);
    let mut map = vec![0u8; width * height];
    for y in 0..height {
        let top = &alpha[rows.low[y] * MODEL_W..][..MODEL_W];
        let bottom = &alpha[rows.high[y] * MODEL_W..][..MODEL_W];
        let ty = rows.fraction[y];
        let target = &mut map[y * width..][..width];
        for (x, slot) in target.iter_mut().enumerate() {
            let (low, high, fx) = (columns.low[x], columns.high[x], columns.fraction[x]);
            let a = top[low] + (top[high] - top[low]) * fx;
            let b = bottom[low] + (bottom[high] - bottom[low]) * fx;
            let value = a + (b - a) * ty;
            *slot = (value * 255.0).round().clamp(0.0, 255.0) as u8;
        }
    }
    map
}

/// A matte written as a frame of the instance's own format: the luma plane
/// with neutral chroma, or the same value in red, green and blue.
fn to_frame(map: &[u8], pix_fmt: PixFmt, width: usize, height: usize, len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    match pix_fmt {
        PixFmt::Yuv420p => {
            out[..width * height].copy_from_slice(map);
            // 128 in both chroma planes is no colour at all.
            out[width * height..].fill(128);
        }
        PixFmt::Rgba => {
            for (pixel, value) in out.as_chunks_mut::<4>().0.iter_mut().zip(map) {
                *pixel = [*value, *value, *value, 255];
            }
        }
    }
    out
}

/// One frame through the graph, however the graph names its input.
fn compute(opened: &Opened, input: &[u8]) -> Result<Vec<(String, Tensor)>, String> {
    let dimensions = [1, 3, MODEL_H as u32, MODEL_W as u32];
    let name = opened.input_name.get();
    let tensor = Tensor::new(&dimensions, TensorType::Fp32, input);
    match opened.context.compute(vec![(name.to_string(), tensor)]) {
        Ok(returned) => Ok(returned),
        // An export whose input is not called what this one calls it. The
        // host takes a position where it takes a name, so the retry names
        // none, and the name that worked is kept for every frame after this
        // one.
        Err(_) if name == INPUT_NAME => {
            opened.input_name.set(INPUT_INDEX);
            let tensor = Tensor::new(&dimensions, TensorType::Fp32, input);
            opened
                .context
                .compute(vec![(INPUT_INDEX.to_string(), tensor)])
                .map_err(|e| failed("compute", &e))
        }
        Err(e) => Err(failed("compute", &e)),
    }
}

/// One frame in, its matte out.
fn run(opened: &Opened, frame: &[u8], len: usize) -> Result<Vec<u8>, String> {
    let input = to_input(frame, opened.pix_fmt, opened.width, opened.height);
    let returned = compute(opened, &input)?;
    let tensors: Vec<Tensor> = returned.into_iter().map(|(_, tensor)| tensor).collect();
    let shapes: Vec<Vec<u32>> = tensors.iter().map(Tensor::dimensions).collect();
    let alpha = output(&shapes)?;
    let map = upscale_alpha(&le_f32s(&tensors[alpha].data()), opened.width, opened.height);
    Ok(to_frame(
        &map,
        opened.pix_fmt,
        opened.width,
        opened.height,
        len,
    ))
}

struct Matte;

impl Guest for Matte {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "matte".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                rows_schema: String::new(),
                pixel_formats: vec!["yuv420p".to_string(), "rgba".to_string()],
                sample_formats: vec![],
                sample_rates: vec![],
                channel_counts: vec![],
                rows_language: vec![],
            },
            window: 1,
            stride: 1,
            pure: true,
            one_to_one: true,
            reads_rows: false,
            forwards_rows: false,
            inputs: 1,
        }
    }

    fn init(format: Format, _stream_info: StreamInfo, params: String) -> Result<(), String> {
        let Format::Video(video) = format else {
            return Err("matte reads frames, and this stream is audio".to_string());
        };
        let pix_fmt = PixFmt::parse(&video.pix_fmt)?;
        validate_params(&params)?;

        // The graph is loaded once per instance, and the session built once:
        // the first frame is what a provider picks its kernels on, and every
        // frame after it reuses them.
        let graph =
            load_by_name(MODEL).map_err(|e| failed(&format!("load-by-name({MODEL:?})"), &e))?;
        let context = graph
            .init_execution_context()
            .map_err(|e| failed("init-execution-context", &e))?;

        OPENED.with(|o| {
            *o.borrow_mut() = Some(Opened {
                width: video.width as usize,
                height: video.height as usize,
                pix_fmt,
                input_name: Cell::new(INPUT_NAME),
                context,
                _graph: graph,
            });
        });
        Ok(())
    }

    fn set_params(params: String) -> Result<(), String> {
        validate_params(&params)
    }

    fn process(window: &InWindow, _trailing: Vec<String>, _last: bool) -> Processed {
        // The final call carries nothing: window and stride are 1, so no
        // frame is ever left over.
        let mut out = Vec::with_capacity(window.len() as usize);
        OPENED.with(|opened| {
            let borrowed = opened.borrow();
            let opened = borrowed
                .as_ref()
                .expect("init loads the graph before any frame arrives");
            for i in 0..window.len() {
                let frame = window.fetch(i);
                match run(opened, &frame, frame.len()) {
                    Ok(map) => out.push(OutFrame {
                        pts: window.pts(i),
                        frame: FramePayload::New(map),
                        rows: vec![],
                    }),
                    // `process` has no way to say no, so a graph that failed
                    // mid-stream stops the run rather than passing a frame
                    // off as a matte.
                    Err(message) => panic!("{message}"),
                }
            }
        });
        Processed {
            frames: out,
            trailing: vec![],
        }
    }
}

export!(Matte);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_identity_resize_reads_each_sample_squarely() {
        let taps = Taps::build(4, 4);
        assert_eq!(taps.low, vec![0, 1, 2, 3]);
        assert_eq!(taps.high, vec![1, 2, 3, 3]);
        assert!(taps.fraction.iter().all(|f| *f == 0.0));
    }

    #[test]
    fn a_two_to_one_shrink_reads_between_its_pairs() {
        // Destination 0 sits at source 0.5: halfway between samples 0 and 1.
        let taps = Taps::build(2, 4);
        assert_eq!((taps.low[0], taps.high[0]), (0, 1));
        assert!((taps.fraction[0] - 0.5).abs() < 1e-6);
        assert_eq!((taps.low[1], taps.high[1]), (2, 3));
        assert!((taps.fraction[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn the_input_is_the_models_own_plane_count_and_size() {
        let (width, height) = (8usize, 8usize);
        let frame = vec![0u8; width * height * 4];
        let bytes = to_input(&frame, PixFmt::Rgba, width, height);
        assert_eq!(bytes.len(), 3 * MODEL_H * MODEL_W * 4);
    }

    #[test]
    fn a_flat_frame_stays_flat_and_lands_in_zero_to_one() {
        // Every pixel 51 in every channel: the stretch cannot invent detail,
        // and 51/255 is exactly 0.2.
        let (width, height) = (16usize, 16usize);
        let frame: Vec<u8> = std::iter::repeat([51, 51, 51, 255])
            .take(width * height)
            .flatten()
            .collect();
        let bytes = to_input(&frame, PixFmt::Rgba, width, height);
        let (words, _) = bytes.as_chunks::<4>();
        for word in words {
            let value = f32::from_le_bytes(*word);
            assert!((value - 0.2).abs() < 1e-6, "got {value}");
        }
    }

    #[test]
    fn neutral_chroma_yuv_is_grayscale_rgb() {
        let (width, height) = (4usize, 4usize);
        let mut frame = vec![100u8; width * height];
        frame.extend(vec![128u8; 2 * 2 * 2]);
        let mut rgb = vec![0f32; width * 3];
        row_to_rgb(&frame, PixFmt::Yuv420p, width, height, 0, &mut rgb);
        for x in 0..width {
            assert!((rgb[x] - 100.0).abs() < 0.01, "red is the luma");
            assert!((rgb[width + x] - 100.0).abs() < 0.01);
            assert!((rgb[2 * width + x] - 100.0).abs() < 0.01);
        }
    }

    #[test]
    fn a_flat_alpha_upscales_flat_and_rounds_to_full() {
        let alpha = vec![1.0f32; MODEL_W * MODEL_H];
        let map = upscale_alpha(&alpha, 33, 17);
        assert!(map.iter().all(|v| *v == 255));
        let none = vec![0.0f32; MODEL_W * MODEL_H];
        assert!(upscale_alpha(&none, 33, 17).iter().all(|v| *v == 0));
    }

    #[test]
    fn an_alpha_edge_upscales_as_a_monotonic_ramp() {
        // Left half 0, right half 1: each output row must never step down.
        let mut alpha = vec![0.0f32; MODEL_W * MODEL_H];
        for row in alpha.chunks_exact_mut(MODEL_W) {
            for value in &mut row[MODEL_W / 2..] {
                *value = 1.0;
            }
        }
        let width = 640;
        let map = upscale_alpha(&alpha, width, 360);
        let row = &map[0..width];
        assert!(row.windows(2).all(|pair| pair[0] <= pair[1]));
        assert_eq!(row[0], 0);
        assert_eq!(row[width - 1], 255);
    }

    #[test]
    fn a_matte_writes_neutral_chroma_and_opaque_alpha() {
        let map = vec![255u8; 4 * 4];
        let yuv = to_frame(&map, PixFmt::Yuv420p, 4, 4, 4 * 4 + 2 * 2 * 2);
        assert!(yuv[..16].iter().all(|v| *v == 255), "the luma is the matte");
        assert!(
            yuv[16..].iter().all(|v| *v == 128),
            "and the chroma is neutral"
        );

        let rgba = to_frame(&map, PixFmt::Rgba, 4, 4, 4 * 4 * 4);
        let (pixels, _) = rgba.as_chunks::<4>();
        for pixel in pixels {
            assert_eq!(
                *pixel,
                [255, 255, 255, 255],
                "equal in every channel, and opaque"
            );
        }
    }

    #[test]
    fn the_alpha_tensor_is_found_by_shape() {
        assert_eq!(output(&[vec![1, 1, 192, 320]]).expect("found"), 0);
        // Beside something else, in either order.
        assert_eq!(
            output(&[vec![1, 4, 96, 160], vec![1, 1, 192, 320]]).expect("found"),
            1
        );
        let error = output(&[vec![1, 1, 320, 192]]).expect_err("transposed is not the alpha");
        assert!(error.starts_with("matte: "), "{error}");
    }

    #[test]
    fn params_are_refused_because_there_are_none() {
        assert!(validate_params("").is_ok());
        assert!(validate_params("{}").is_ok());
        let error = validate_params(r#"{"conf":0.5}"#).expect_err("no params exist");
        assert!(error.contains("matte takes no params"), "{error}");
    }
}
