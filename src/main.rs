use clap::Parser;
use opencv::{core::Scalar, imgcodecs, prelude::*};
use smiletrack_rs::{
    config::Config,
    detection::Detector,
    utils::{draw_box, put_text},
};

/// Simple demo for SMILEtrack end-to-end detection on a single image.
#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    /// Path to the config JSON file
    #[arg(long)]
    config: String,
    /// Input image path
    #[arg(long)]
    input: String,
    /// Output image path
    #[arg(long, default_value = "output.jpg")]
    output: String,
}

fn main() -> anyhow::Result<()> {
    // 1. Parse CLI
    let args = Args::parse();

    // 2. Load config and init detector
    let cfg = Config::from_file(&args.config)?;
    let detector = Detector::new(&cfg.model_path, &cfg.device)?;

    // 3. Read image
    let mut img = imgcodecs::imread(&args.input, imgcodecs::IMREAD_COLOR)?;

    // 4. Detect & draw boxes
    let dets = detector.detect(&img)?;
    for det in &dets {
        let [x1, y1, x2, y2] = [
            det.bbox[0] as i32,
            det.bbox[1] as i32,
            det.bbox[2] as i32,
            det.bbox[3] as i32,
        ];
        draw_box(&mut img, [x1, y1, x2, y2], Scalar::new(0.0, 255.0, 0.0, 0.0), 2)?;
        put_text(
            &mut img,
            &format!("{:.2}", det.score),
            (x1, y1 - 5),
            Scalar::new(0.0, 255.0, 0.0, 0.0),
            0.5,
            1,
        )?;
    }

    // 5. Save output image
    imgcodecs::imwrite(&args.output, &img, &opencv::types::VectorOfint::new())?;
    Ok(())
}