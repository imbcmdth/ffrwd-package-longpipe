//! Whole-frame matting: every frame leaves as a grayscale alpha of its
//! foreground - 255 where the subject owns the pixel, 0 where the background
//! does, and the values between are the blend.
//!
//! The graph is longpipe's fused temporal xl export, run through `wasi:nn`
//! at its fixed 1280x768 canvas. One compute takes the frame beside the
//! previous frame's state - that frame at the flow half's base resolution,
//! its four encoder taps, and the stabilizer carrier - and answers the
//! stabilized alpha along with that state refreshed. The module holds the
//! state between calls and feeds it back, so each matte is steadied against
//! the frame before it. A call therefore depends on more than it was
//! handed, and the module says so: it is impure, hosted one call at a time,
//! in order.
//!
//! The first frame of an instance has no previous frame. Its state is all
//! zeros, its matte is the graph's raw alpha, and that alpha seeds the
//! carrier beside a zero envelope - the cold start the export was built
//! around. A new stream opens a new instance, so a discontinuity starts
//! fresh.
//!
//! The frame is stretch-resized to the canvas - no letterbox, no crop, each
//! axis by its own ratio - as RGB rescaled to 0..1 with no further
//! normalization, and the alpha that comes back is resized bilinearly onto
//! the frame's own geometry. The module never opens a file - the host binds
//! the graph to a name with `-nn matte=<path>` and this module asks for that
//! name and nothing else.
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

use std::cell::RefCell;

use exports::ffrwd::av::window_filter::{
    Format, FramePayload, Guest, InWindow, Meta, OutFrame, Processed, StreamInfo, WindowMeta,
};
use wasi::nn::graph::{load_by_name, Graph};
use wasi::nn::inference::GraphExecutionContext;
use wasi::nn::tensor::{Tensor, TensorType};

/// The name the host binds the graph to. `-nn matte=<path>`.
const MODEL: &str = "matte";

/// The canvas the graph is exported at, fixed: the frame input, both alphas
/// and the stabilizer carrier all share it.
const CANVAS_W: usize = 1280;
const CANVAS_H: usize = 768;

/// What the export calls its frame input.
const FRAME_IN: &str = "frame";

/// The cross-frame inputs, in the export's order: the previous frame at the
/// flow half's base resolution, its four encoder taps, and the stabilizer
/// carrier - alpha beside envelope.
const STATE_IN: [&str; 6] = [
    "prev_frame_base",
    "prev_tap0",
    "prev_tap1",
    "prev_tap2",
    "prev_tap3",
    "prev_stab",
];

/// The refreshed state the graph answers with, matching `STATE_IN` slot for
/// slot. The carrier is not here: a warm frame reads `stab_state`, and the
/// first frame seeds it from the raw alpha instead.
const STATE_OUT: [&str; 5] = [
    "cur_frame_base",
    "cur_tap0",
    "cur_tap1",
    "cur_tap2",
    "cur_tap3",
];

/// Each state tensor's shape, in `STATE_IN` order, shared by the output that
/// refreshes it.
const STATE_DIMS: [[u32; 4]; 6] = [
    [1, 3, 192, 320],
    [1, 32, 48, 80],
    [1, 48, 24, 40],
    [1, 136, 12, 20],
    [1, 384, 6, 10],
    [1, 2, CANVAS_H as u32, CANVAS_W as u32],
];

/// The alphas the graph answers with: stabilized for a warm frame, raw for
/// the first, which has nothing to stabilize against.
const ALPHA_STAB: &str = "alpha_stab";
const ALPHA_RAW: &str = "alpha_raw";

/// The refreshed carrier a warm frame reads back.
const STAB_STATE: &str = "stab_state";

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;

/// A shape's length as the little-endian fp32 bytes it travels in.
fn bytes_of(dims: &[u32]) -> usize {
    dims.iter().map(|d| *d as usize).product::<usize>() * 4
}

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

/// The tensors carried from one frame to the next, as the bytes they travel
/// in, in `STATE_IN` order. `warm` says whether any frame has run: until one
/// has, the tensors are zeros and the carrier has nothing to steady against.
struct State {
    tensors: [Vec<u8>; 6],
    warm: bool,
}

impl State {
    /// Every tensor zero - an fp32 zero is four zero bytes.
    fn cold() -> State {
        State {
            tensors: STATE_DIMS.map(|dims| vec![0u8; bytes_of(&dims)]),
            warm: false,
        }
    }
}

/// What `init` settled, plus the graph it loaded and the state the stream
/// has reached.
struct Opened {
    width: usize,
    height: usize,
    pix_fmt: PixFmt,
    /// Carried across `process` calls and fed back into every compute.
    state: State,
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

/// One frame row resized to the canvas's columns, channel by channel.
fn resize_rgb_row(rgb: &[f32], columns: &Taps, width: usize, out: &mut [f32]) {
    for channel in 0..3 {
        let source = &rgb[channel * width..(channel + 1) * width];
        let target = &mut out[channel * CANVAS_W..(channel + 1) * CANVAS_W];
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

/// The frame stretched to the 1280x768 canvas - each axis by its own ratio,
/// no letterbox, the contract the weights shipped with - and laid out as the
/// planar fp32 tensor the graph expects: red, green and blue in turn, each
/// rescaled to 0..1 and nothing else.
fn to_input(frame: &[u8], pix_fmt: PixFmt, width: usize, height: usize) -> Vec<u8> {
    let plane = CANVAS_W * CANVAS_H;
    let mut planes = vec![0f32; plane * 3];

    let columns = Taps::build(CANVAS_W, width);
    let rows = Taps::build(CANVAS_H, height);
    let mut rgb = vec![0f32; width * 3];
    let mut top = vec![0f32; CANVAS_W * 3];
    let mut bottom = vec![0f32; CANVAS_W * 3];

    for my in 0..CANVAS_H {
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
            let a = &top[channel * CANVAS_W..(channel + 1) * CANVAS_W];
            let b = &bottom[channel * CANVAS_W..(channel + 1) * CANVAS_W];
            let target = &mut planes[channel * plane + my * CANVAS_W..][..CANVAS_W];
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

/// The named tensor out of what a compute answered, checked against the byte
/// length its shape fixes: these bytes are fed back next frame, so a wrong
/// size is refused here rather than corrupting every frame after it.
fn take(outputs: &mut Vec<(String, Vec<u8>)>, name: &str, len: usize) -> Result<Vec<u8>, String> {
    let Some(at) = outputs.iter().position(|(n, _)| n == name) else {
        let unclaimed: Vec<&str> = outputs.iter().map(|(n, _)| n.as_str()).collect();
        return Err(format!(
            "matte: the graph answered without {name:?} (unclaimed: {unclaimed:?}); \
             this module wants longpipe's fused temporal export"
        ));
    };
    let (_, bytes) = outputs.swap_remove(at);
    if bytes.len() != len {
        return Err(format!(
            "matte: {name} came back as {} bytes where its shape fixes {len}",
            bytes.len()
        ));
    }
    Ok(bytes)
}

/// One compute's answer folded into the state, returning the alpha this
/// frame shows. A warm frame shows `alpha_stab` and carries `stab_state`;
/// the first frame shows `alpha_raw` and seeds the carrier with it beside a
/// zero envelope, the way the export's cold start is defined.
fn advance(state: &mut State, mut outputs: Vec<(String, Vec<u8>)>) -> Result<Vec<u8>, String> {
    let plane = CANVAS_W * CANVAS_H * 4;
    let raw = take(&mut outputs, ALPHA_RAW, plane)?;
    for (i, name) in STATE_OUT.iter().enumerate() {
        state.tensors[i] = take(&mut outputs, name, bytes_of(&STATE_DIMS[i]))?;
    }
    if state.warm {
        state.tensors[5] = take(&mut outputs, STAB_STATE, 2 * plane)?;
        take(&mut outputs, ALPHA_STAB, plane)
    } else {
        state.warm = true;
        let mut carrier = raw.clone();
        carrier.resize(2 * plane, 0);
        state.tensors[5] = carrier;
        Ok(raw)
    }
}

/// The graph's alpha brought onto the frame's own geometry, bilinearly, as
/// 8-bit gray.
fn upscale_alpha(alpha: &[f32], width: usize, height: usize) -> Vec<u8> {
    let columns = Taps::build(width, CANVAS_W);
    let rows = Taps::build(height, CANVAS_H);
    let mut map = vec![0u8; width * height];
    for y in 0..height {
        let top = &alpha[rows.low[y] * CANVAS_W..][..CANVAS_W];
        let bottom = &alpha[rows.high[y] * CANVAS_W..][..CANVAS_W];
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

/// One frame through the graph: the frame stretched to canvas beside the
/// carried state, the state refreshed from the answer, the alpha brought
/// onto the frame's own geometry.
fn run(opened: &mut Opened, frame: &[u8], len: usize) -> Result<Vec<u8>, String> {
    let input = to_input(frame, opened.pix_fmt, opened.width, opened.height);
    let mut feeds = Vec::with_capacity(1 + STATE_IN.len());
    let frame_dims = [1, 3, CANVAS_H as u32, CANVAS_W as u32];
    feeds.push((
        FRAME_IN.to_string(),
        Tensor::new(&frame_dims, TensorType::Fp32, &input),
    ));
    for ((name, dims), bytes) in STATE_IN.iter().zip(&STATE_DIMS).zip(&opened.state.tensors) {
        feeds.push((name.to_string(), Tensor::new(dims, TensorType::Fp32, bytes)));
    }
    let returned = opened
        .context
        .compute(feeds)
        .map_err(|e| failed("compute", &e))?;
    let outputs: Vec<(String, Vec<u8>)> = returned
        .into_iter()
        .map(|(name, tensor)| (name, tensor.data()))
        .collect();
    let alpha = advance(&mut opened.state, outputs)?;
    let map = upscale_alpha(&le_f32s(&alpha), opened.width, opened.height);
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
            // Each call reads the state the one before it left, so calls
            // happen one at a time, in order.
            pure: false,
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
                state: State::cold(),
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
            let mut borrowed = opened.borrow_mut();
            let opened = borrowed
                .as_mut()
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
    fn the_input_is_the_canvas_plane_count_and_size() {
        let (width, height) = (8usize, 8usize);
        let frame = vec![0u8; width * height * 4];
        let bytes = to_input(&frame, PixFmt::Rgba, width, height);
        assert_eq!(bytes.len(), 3 * CANVAS_H * CANVAS_W * 4);
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
        let alpha = vec![1.0f32; CANVAS_W * CANVAS_H];
        let map = upscale_alpha(&alpha, 33, 17);
        assert!(map.iter().all(|v| *v == 255));
        let none = vec![0.0f32; CANVAS_W * CANVAS_H];
        assert!(upscale_alpha(&none, 33, 17).iter().all(|v| *v == 0));
    }

    #[test]
    fn an_alpha_edge_upscales_as_a_monotonic_ramp() {
        // Left half 0, right half 1: each output row must never step down.
        let mut alpha = vec![0.0f32; CANVAS_W * CANVAS_H];
        for row in alpha.chunks_exact_mut(CANVAS_W) {
            for value in &mut row[CANVAS_W / 2..] {
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

    /// The full answer a compute returns, each tensor filled with its own
    /// byte so a test can see which one landed where. `flow` rides along
    /// unclaimed, as it does in the real answer.
    fn answered() -> Vec<(String, Vec<u8>)> {
        let plane = CANVAS_W * CANVAS_H * 4;
        let mut outputs = vec![
            (ALPHA_STAB.to_string(), vec![2u8; plane]),
            (ALPHA_RAW.to_string(), vec![1u8; plane]),
            ("flow".to_string(), vec![9u8; 2 * 48 * 80 * 4]),
            (STAB_STATE.to_string(), vec![3u8; 2 * plane]),
        ];
        for (i, name) in STATE_OUT.iter().enumerate() {
            outputs.push((name.to_string(), vec![10 + i as u8; bytes_of(&STATE_DIMS[i])]));
        }
        outputs
    }

    #[test]
    fn a_cold_state_is_zeros_in_the_exports_shapes() {
        let state = State::cold();
        assert!(!state.warm);
        for (tensor, dims) in state.tensors.iter().zip(&STATE_DIMS) {
            assert_eq!(tensor.len(), bytes_of(dims));
            assert!(tensor.iter().all(|b| *b == 0));
        }
    }

    #[test]
    fn the_first_frame_shows_the_raw_matte_and_seeds_the_carrier() {
        let mut state = State::cold();
        let alpha = advance(&mut state, answered()).expect("advances");
        assert!(
            alpha.iter().all(|b| *b == 1),
            "the raw alpha, not the stabilized one"
        );
        assert!(state.warm);
        let plane = CANVAS_W * CANVAS_H * 4;
        assert!(
            state.tensors[5][..plane].iter().all(|b| *b == 1),
            "the carrier's alpha lane is the raw matte"
        );
        assert!(
            state.tensors[5][plane..].iter().all(|b| *b == 0),
            "beside a zero envelope"
        );
        for (i, tensor) in state.tensors[..5].iter().enumerate() {
            assert!(
                tensor.iter().all(|b| *b == 10 + i as u8),
                "state slot {i} is the refreshed tensor"
            );
        }
    }

    #[test]
    fn a_warm_frame_shows_the_stabilized_matte_and_reads_the_carrier_back() {
        let mut state = State::cold();
        advance(&mut state, answered()).expect("the cold frame");
        let alpha = advance(&mut state, answered()).expect("the warm one");
        assert!(alpha.iter().all(|b| *b == 2), "the stabilized alpha");
        assert!(
            state.tensors[5].iter().all(|b| *b == 3),
            "the carrier is stab_state now"
        );
    }

    #[test]
    fn an_answer_missing_a_state_tensor_is_refused_by_name() {
        let mut state = State::cold();
        let outputs: Vec<_> = answered()
            .into_iter()
            .filter(|(name, _)| name != "cur_tap2")
            .collect();
        let error = advance(&mut state, outputs).expect_err("cur_tap2 is gone");
        assert!(error.contains("cur_tap2"), "{error}");
    }

    #[test]
    fn a_state_tensor_of_the_wrong_size_is_refused() {
        let mut state = State::cold();
        let mut outputs = answered();
        let at = outputs
            .iter()
            .position(|(name, _)| name == "cur_frame_base")
            .expect("present");
        outputs[at].1.truncate(16);
        let error = advance(&mut state, outputs).expect_err("truncated");
        assert!(error.contains("cur_frame_base"), "{error}");
    }

    #[test]
    fn params_are_refused_because_there_are_none() {
        assert!(validate_params("").is_ok());
        assert!(validate_params("{}").is_ok());
        let error = validate_params(r#"{"conf":0.5}"#).expect_err("no params exist");
        assert!(error.contains("matte takes no params"), "{error}");
    }
}
