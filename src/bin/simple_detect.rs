use clap::{Arg, Command};
use opencv::{imgcodecs, prelude::*};
use smiletrack::simple_detector::{SimpleDetector, SimpleFrameResult, SimpleTrack};
use std::fs::File;
use std::io::Write;
use anyhow::Result;

fn main() -> Result<()> {
    // Parse command line arguments
    let matches = Command::new("Simple Detector")
        .version("0.1.0")
        .about("Simple detector that outputs detection results similar to Python")
        .arg(
            Arg::new("input")
                .short('i')
                .long("input")
                .value_name("INPUT")
                .help("Input image or video file")
                .required(true),
        )
        .arg(
            Arg::new("weights")
                .short('w')
                .long("weights")
                .value_name("WEIGHTS")
                .help("Path to model weights file")
                .required(true),
        )
        .arg(
            Arg::new("conf_threshold")
                .short('c')
                .long("conf_threshold")
                .value_name("CONF")
                .help("Confidence threshold")
                .default_value("0.25"),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("OUTPUT")
                .help("Output JSON file")
                .default_value("./detections.json"),
        )
        .get_matches();

    let input_path = matches.get_one::<String>("input").unwrap();
    let weights_path = matches.get_one::<String>("weights").unwrap();
    let conf_threshold = matches
        .get_one::<String>("conf_threshold")
        .unwrap()
        .parse::<f32>()
        .unwrap_or(0.25);
    let output_path = matches.get_one::<String>("output").unwrap();

    println!("Loading model from: {}", weights_path);
    println!("Using confidence threshold: {}", conf_threshold);

    // Initialize the simple detector
    let detector = SimpleDetector::new(
        weights_path,
        "cpu",
        (640, 640),
        conf_threshold,
        0.45,
    )?;

    println!("Processing input: {}", input_path);

    // Read input frame
    let frame = imgcodecs::imread(input_path, imgcodecs::IMREAD_COLOR)?;
    if frame.rows() == 0 || frame.cols() == 0 {
        println!("Error: Could not read input image: {}", input_path);
        return Ok(());
    }

    // Process frame
    let frame_result = detector.process_frame(&frame, 0)?;

    // Add track IDs to create tracks
    let output_result = SimpleFrameResult {
        frame_id: frame_result.frame_id,
        detections: frame_result.detections.clone(),
        tracks: create_tracks_from_detections(&frame_result.detections),
    };

    // Save results to JSON
    let json_str = serde_json::to_string_pretty(&vec![output_result])?;
    
    println!("Saving results to: {}", output_path);
    let mut file = File::create(output_path)?;
    file.write_all(json_str.as_bytes())?;

    println!("Done!");
    Ok(())
}

// Helper function to create tracks from detections
fn create_tracks_from_detections(detections: &[smiletrack::simple_detector::SimpleDetection]) -> Vec<SimpleTrack> {
    let mut tracks = Vec::new();
    
    for (id, detection) in detections.iter().enumerate() {
        let track = SimpleTrack {
            track_id: (id + 1) as i32,  // Start track IDs from 1
            bbox: detection.bbox,
            confidence: detection.confidence,
            class_id: detection.class_id,
            class_name: detection.class_name.clone(),
        };
        
        tracks.push(track);
    }
    
    tracks
} 