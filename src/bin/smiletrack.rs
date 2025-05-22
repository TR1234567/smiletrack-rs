use clap::Parser;
use opencv::{
    prelude::*,
    videoio::{self, VideoCapture, VideoWriter, CAP_ANY},
    highgui,
    core::Size,
    imgcodecs,
};
use std::{path::PathBuf, fs};
use smiletrack::{Config, Detector, SMILEtrack, visualization, STrack};
use smiletrack::detection::Detection;
use std::fs::File;
use std::io::Write;
use serde::{Serialize, Deserialize};
use serde_json;
use anyhow;
use std::path::Path;
use std::sync::Arc;

#[derive(Parser)]
#[command(
    name = "smiletrack",
    about = "Rust implementation of SMILEtrack for MOT tracking",
    version = "0.1.0"
)]
struct Args {
    /// Path to video file or image
    #[arg(short, long, required = true)]
    input: PathBuf,

    /// Output path (directory for frames or video file)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Path to configuration file
    #[arg(short, long, default_value = "config.json")]
    config: Option<PathBuf>,

    /// Path to model weights
    #[arg(short, long)]
    weights: Option<PathBuf>,

    /// Enable visualization
    #[arg(short, long)]
    visualize: bool,

    /// Frames per second (for video output)
    #[arg(long, default_value_t = 30.0)]
    fps: f64,
}

#[derive(Serialize, Deserialize, Debug)]
struct YoloAnnotation {
    frame: String,
    annotations: Vec<BoundingBox>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct BoundingBox {
    class_id: i32,
    confidence: f32,
    x_center: f32,
    y_center: f32,
    width: f32,
    height: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct DetectionLog {
    bbox: Vec<f32>,         // [x, y, w, h] format
    confidence: f32,
    class_id: i32,
    class_name: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct TrackLog {
    track_id: u32,
    bbox: Vec<f32>,         // [x, y, w, h] format
    confidence: f32,
    class_id: i32,
    class_name: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct FrameLog {
    frame_id: i32,
    detections: Vec<DetectionLog>,
    tracks: Vec<TrackLog>,
}

struct ProcessingState {
    detector: Detector,
    tracker: SMILEtrack,
    writer: Option<VideoWriter>,
    annotations: Vec<YoloAnnotation>,
    annotation_path: Option<PathBuf>,
    vis_output_dir: Option<PathBuf>,
    window_name: String,
    show_visualization: bool,
    show_detections: bool,
    tracking_log: Vec<FrameLog>,
    tracking_log_path: Option<PathBuf>,
}

impl ProcessingState {
    fn process_frame(&mut self, frame: &Mat, frame_path: Option<&str>, frame_id: i32, fps: f64) -> Result<bool, Box<dyn std::error::Error>> {
        // Run detection
        let detections = self.detector.detect(frame)?;
        println!("{} detections found", detections.len());

        // Print high confidence detections
        let high_conf_dets: Vec<_> = detections.iter()
            .filter(|det| det.confidence >= 0.25)
            .collect();
        
        println!("{} high confidence detections", high_conf_dets.len());
        
        for det in high_conf_dets.iter().take(5) {  // Show first 5 high confidence detections
            let tlwh = det.tlwh();
            println!("High score detection: class={}, score={:.3}, box=[{:.1}, {:.1}, {:.1}, {:.1}]", 
                det.class_id, det.confidence, tlwh[0], tlwh[1], tlwh[2], tlwh[3]);
        }

        // Update tracks
        println!("Updating tracks...");
        self.tracker.update(&detections, frame, frame_id)?;
        
        // Get tracks that are activated
        let tracks = self.tracker.tracks();
        let activated_tracks: Vec<STrack> = tracks.iter()
            .filter(|t| t.is_activated())
            .cloned()
            .collect();

        println!("{} tracks are activated", activated_tracks.len());
        
        // Log tracking details for comparison with Python
        self.log_tracking_details(frame_id, &detections, &activated_tracks)?;

        // If annotation path is provided and we have a frame path, save annotations
        if let (Some(frame_path_str), true) = (frame_path, self.annotation_path.is_some()) {
            let img_width = frame.cols() as f32;
            let img_height = frame.rows() as f32;
            
            // Convert detections to YOLO format
            let mut yolo_boxes = Vec::new();
            for det in &detections {
                let tlwh = det.tlwh();
                
                // YOLO format: x_center, y_center, width, height (normalized 0-1)
                let x_center = (tlwh[0] + tlwh[2] / 2.0) / img_width;
                let y_center = (tlwh[1] + tlwh[3] / 2.0) / img_height;
                let width = tlwh[2] / img_width;
                let height = tlwh[3] / img_height;
                
                yolo_boxes.push(BoundingBox {
                    class_id: det.class_id,
                    confidence: det.confidence,
                    x_center,
                    y_center,
                    width,
                    height,
                });
            }
            
            // Create annotation for this frame
            let annotation = YoloAnnotation {
                frame: frame_path_str.to_string(),
                annotations: yolo_boxes,
            };
            
            // Add to annotation collection
            self.annotations.push(annotation);
        }

        // Create visualization with tracking results
        let mut output_frame = frame.clone();
        
        // Draw frame information - number of tracks
        let track_count_text = format!("Total Tracked IDs: {}", activated_tracks.len());
        visualization::draw_text(&mut output_frame, &track_count_text, 20, 30, 0.7, (0, 255, 0))?;
        
        // Draw frame info - frame number, fps
        visualization::draw_frame_info(&mut output_frame, frame_id, fps)?;

        // Draw detections if requested
        if self.show_detections {
            visualization::draw_detections(&mut output_frame, &detections)?;
        }

        // Draw tracks
        visualization::draw_tracks(&mut output_frame, &activated_tracks)?;
        
        // Save visualization frame if output directory is provided
        if let Some(vis_dir) = &self.vis_output_dir {
            // Make sure vis_dir is a directory, not a file
            if vis_dir.exists() && !vis_dir.is_dir() {
                println!("Warning: Output path {:?} is a file, not a directory. Skipping visualization output.", vis_dir);
            } else {
                // Create the directory if it doesn't exist
                if !vis_dir.exists() {
                    println!("Creating output directory: {:?}", vis_dir);
                    fs::create_dir_all(vis_dir).map_err(|e| {
                        println!("Failed to create directory: {}", e);
                        e
                    })?;
                }
                
                let output_filename = if let Some(frame_path_str) = frame_path {
                    // For image sequence, use the original filename with a prefix
                    let original_path = PathBuf::from(frame_path_str);
                    let filename = original_path.file_name().unwrap().to_string_lossy().to_string();
                    format!("vis_{}", filename)
                } else {
                    // For video, use frame number
                    format!("frame_{:06}.jpg", frame_id)
                };
                
                let output_path = vis_dir.join(output_filename);
                println!("Writing output to: {:?}", output_path);
                imgcodecs::imwrite(
                    &output_path.to_string_lossy(),
                    &output_frame,
                    &opencv::core::Vector::new()
                )?;
            }
        }
        
        // Show visualization if requested
        if self.show_visualization {
            highgui::imshow(&self.window_name, &output_frame)?;
            let key = highgui::wait_key(1)?;
            if key == 27 {  // ESC key
                println!("\nTracking interrupted by user.");
                return Ok(false);
            }
        }

        // Write to video if requested
        if let Some(writer) = &mut self.writer {
            writer.write(&output_frame)?;
        }

        Ok(true)
    }
    
    fn log_tracking_details(&mut self, frame_id: i32, detections: &[Detection], tracks: &[STrack]) -> Result<(), Box<dyn std::error::Error>> {
        // Skip if no logging path is set
        if self.tracking_log_path.is_none() {
            return Ok(());
        }
        
        // Log ALL detections without filtering
        let mut detection_logs = Vec::new();
        
        println!("Logging all {} detections for comparison", detections.len());
        
        for det in detections {
            let tlwh = det.tlwh();
            let class_name = match det.class_id {
                0 => "person".to_string(),
                1 => "bicycle".to_string(),
                2 => "car".to_string(),
                3 => "motorcycle".to_string(),
                5 => "bus".to_string(),
                7 => "truck".to_string(),
                15 => "cat".to_string(),
                16 => "dog".to_string(),
                _ => format!("class_{}", det.class_id),
            };
            
            // Print each detection for debugging
            println!("Detection: class={} ({}), conf={:.3}, bbox=[{:.1}, {:.1}, {:.1}, {:.1}]",
                class_name, det.class_id, det.confidence, tlwh[0], tlwh[1], tlwh[2], tlwh[3]);
            
            detection_logs.push(DetectionLog {
                bbox: vec![tlwh[0], tlwh[1], tlwh[2], tlwh[3]],
                confidence: det.confidence,
                class_id: det.class_id,
                class_name,
            });
        }
        
        // Log ALL tracks, not just activated ones for debugging
        let mut track_logs = Vec::new();
        
        println!("Logging all {} tracks for comparison", tracks.len());
        
        for track in tracks {
            let tlwh = track.tlwh().clone();
            let class_name = match track.class_id {
                0 => "person".to_string(),
                1 => "bicycle".to_string(),
                2 => "car".to_string(),
                3 => "motorcycle".to_string(),
                5 => "bus".to_string(),
                7 => "truck".to_string(),
                15 => "cat".to_string(),
                16 => "dog".to_string(),
                _ => format!("class_{}", track.class_id),
            };
            
            // Print each track for debugging
            println!("Track: id={}, class={} ({}), conf={:.3}, bbox=[{:.1}, {:.1}, {:.1}, {:.1}], activated={}",
                track.track_id(), class_name, track.class_id, track.score, 
                tlwh[0], tlwh[1], tlwh[2], tlwh[3], track.is_activated());
            
            track_logs.push(TrackLog {
                track_id: track.track_id(),
                bbox: vec![tlwh[0], tlwh[1], tlwh[2], tlwh[3]],
                confidence: track.score,
                class_id: track.class_id,
                class_name,
            });
        }
        
        // Create frame log
        let frame_log = FrameLog {
            frame_id,
            detections: detection_logs,
            tracks: track_logs,
        };
        
        // Add to tracking log
        self.tracking_log.push(frame_log);
        
        // Write to file (write the entire log each time to handle crashes)
        if let Some(path) = &self.tracking_log_path {
            let json = serde_json::to_string_pretty(&self.tracking_log)?;
            let mut file = File::create(path)?;
            file.write_all(json.as_bytes())?;
            println!("Updated tracking log saved to {:?}", path);
        }
        
        Ok(())
    }
    
    // Save annotations to JSON file
    fn save_annotations(&self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(path) = &self.annotation_path {
            println!("Saving annotations to {:?}...", path);
            let json = serde_json::to_string_pretty(&self.annotations)?;
            let mut file = File::create(path)?;
            file.write_all(json.as_bytes())?;
            println!("Annotations saved successfully.");
        }
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args = Args::parse();

    // Load config
    println!("Loading configuration from {:?}...", args.config.as_deref().unwrap_or(&PathBuf::from("config.json")));
    let mut config = Config::from_file(args.config.as_deref().unwrap_or(&PathBuf::from("config.json")).to_string_lossy().as_ref())?;
    
    // Override config with command line arguments if provided
    if let Some(weights) = &args.weights {
        config.model_path = weights.to_string_lossy().to_string();
    }
    
    // Use a very low threshold to catch everything, we'll filter later for visualization
    config.conf_threshold = 0.001;  // Catch all detections
    
    println!("Initializing detector with weights from {:?}...", config.model_path);
    println!("Using VERY LOW confidence threshold: {}", config.conf_threshold);
    println!("Using track threshold: {}", config.track_high_thresh);
    
    // Initialize detector with specific classes
    let mut detector = Detector::new(
        &config.model_path,
        &config.device,
        (config.input_size[0] as i64, config.input_size[1] as i64),
        config.conf_threshold,
        config.nms_threshold,
    )?;
    
    // Set allowed classes to match Python implementation
    detector.set_classes(vec![0, 1, 2, 3, 5, 7, 15, 16]);
    println!("Detector will only consider classes: [0, 1, 2, 3, 5, 7, 15, 16]");
    println!("These correspond to: person, bicycle, car, motorcycle, bus, truck, cat, dog");
    
    // Initialize tracker (passing FPS for motion model)
    let tracker = SMILEtrack::new(&config, args.fps as f32);
    
    // Check if input is an image or video
    let is_image = match args.input.extension().and_then(|e| e.to_str()) {
        Some(ext) => matches!(ext.to_lowercase().as_str(), "jpg" | "jpeg" | "png" | "bmp"),
        None => false,
    };
    
    // Open input source
    println!("Opening input file {:?}...", args.input);
    
    // Handle single image input
    if is_image {
        println!("Processing single image input...");
        let frame = imgcodecs::imread(&args.input.to_string_lossy(), imgcodecs::IMREAD_COLOR)?;
        if frame.empty() {
            return Err(anyhow::anyhow!("Failed to load image: {:?}", args.input).into());
        }
        
        // Create visualization window if needed
        if args.visualize {
            highgui::named_window("SMILEtrack", highgui::WINDOW_NORMAL)?;
            highgui::resize_window("SMILEtrack", frame.cols(), frame.rows())?;
        }
        
        // For single image input, determine if we're outputting directly to a file or to a directory
        let (vis_output_dir, direct_output_file) = if let Some(output_path) = &args.output {
            let is_image_extension = match output_path.extension().and_then(|e| e.to_str()) {
                Some(ext) => matches!(ext.to_lowercase().as_str(), "jpg" | "jpeg" | "png" | "bmp"),
                None => false,
            };
            
            if is_image_extension {
                // Direct file output - we'll save directly to this file
                (None, Some(output_path.clone()))
            } else {
                // Directory output - we'll save to a file in this directory
                // Ensure directory exists
                if !output_path.exists() || !output_path.is_dir() {
                    println!("Creating output directory: {:?}", output_path);
                    fs::create_dir_all(output_path)?;
                }
                (Some(output_path.clone()), None)
            }
        } else {
            (None, None)
        };
        
        // Setup tracking log path
        let tracking_log_path = if let Some(output_dir) = &vis_output_dir {
            Some(output_dir.join("tracking_details.json"))
        } else if let Some(parent) = direct_output_file.as_ref().and_then(|p| p.parent()) {
            Some(parent.join("tracking_details.json"))
        } else {
            Some(PathBuf::from("tracking_details.json"))
        };
        
        // Create processing state
        let mut processing_state = ProcessingState {
            detector,
            tracker,
            writer: None,
            annotations: Vec::new(),
            annotation_path: None,
            vis_output_dir,
            window_name: "SMILEtrack".to_string(),
            show_visualization: args.visualize,
            show_detections: true, // Show detections for images
            tracking_log: Vec::new(),
            tracking_log_path,
        };
        
        // Process the single image frame
        let frame_path = args.input.to_string_lossy().to_string();
        processing_state.process_frame(&frame, Some(&frame_path), 0, args.fps)?;
        
        // If direct output file is specified, save the result directly
        if let Some(direct_output_path) = direct_output_file {
            println!("Saving final result to {:?}", direct_output_path);
            
            // Create a visualization with tracking results
            let mut output_frame = frame.clone();
            
            // Draw detections and tracks
            let detections = processing_state.detector.detect(&frame)?;
            visualization::draw_detections(&mut output_frame, &detections)?;
            
            let tracks = processing_state.tracker.tracks();
            let activated_tracks: Vec<STrack> = tracks.iter()
                .filter(|t| t.is_activated())
                .cloned()
                .collect();
            visualization::draw_tracks(&mut output_frame, &activated_tracks)?;
            
            // Ensure parent directory exists
            if let Some(parent) = direct_output_path.parent() {
                if !parent.exists() {
                    fs::create_dir_all(parent)?;
                }
            }
            
            // Write the final result
            imgcodecs::imwrite(
                &direct_output_path.to_string_lossy(),
                &output_frame,
                &opencv::core::Vector::new()
            )?;
        }
        
        // Wait for key press if showing visualization
        if args.visualize {
            println!("Press any key to exit...");
            highgui::wait_key(0)?;
        }
    } else {
        // Handle video input
        println!("Processing video input...");
        let mut cap = VideoCapture::from_file(&args.input.to_string_lossy(), videoio::CAP_ANY)?;
        if !cap.is_opened()? {
            return Err(anyhow::anyhow!("Failed to open video file: {:?}", args.input).into());
        }
        
        // Get video properties
        let width = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
        let height = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;
        let total_frames = cap.get(videoio::CAP_PROP_FRAME_COUNT)? as i32;
        let fps = cap.get(videoio::CAP_PROP_FPS)?;
        
        println!("Video properties:");
        println!("  Resolution: {}x{}", width, height);
        println!("  Total frames: {}", total_frames);
        println!("  FPS: {:.2}", fps);
        
        // Setup output writer
        let mut video_writer: Option<VideoWriter> = None;
        let mut vis_output_dir: Option<PathBuf> = None;
        
        if let Some(output_path) = &args.output {
            // Create parent directory if it doesn't exist
            if let Some(parent) = output_path.parent() {
                fs::create_dir_all(parent)?;
            }
            
            if output_path.extension().and_then(|e| e.to_str()) == Some("mp4") {
                // Video output
                println!("Setting up video writer to {:?}", output_path);
                let fourcc = VideoWriter::fourcc('a', 'v', 'c', '1')?;
                let video_writer_obj = VideoWriter::new(
                    &output_path.to_string_lossy(),
                    fourcc,
                    args.fps,
                    Size::new(width, height),
                    true,
                )?;
                
                if !video_writer_obj.is_opened()? {
                    println!("Warning: Failed to open video writer, falling back to image sequence");
                    vis_output_dir = Some(output_path.clone());
                } else {
                    video_writer = Some(video_writer_obj);
                }
            } else {
                // Directory output for frame sequence
                vis_output_dir = Some(output_path.clone());
                if !output_path.exists() {
                    fs::create_dir_all(output_path)?;
                }
            }
        }
        
        // Create visualization window if needed
        if args.visualize {
            highgui::named_window("SMILEtrack", highgui::WINDOW_NORMAL)?;
            highgui::resize_window("SMILEtrack", width, height)?;
        }
        
        // Setup tracking log path
        let tracking_log_path = if let Some(output_dir) = &vis_output_dir {
            Some(output_dir.join("tracking_details.json"))
        } else {
            Some(PathBuf::from("tracking_details.json"))
        };
        
        // Create processing state
        let mut processing_state = ProcessingState {
            detector,
            tracker,
            writer: video_writer,
            annotations: Vec::new(),
            annotation_path: None,
            vis_output_dir,
            window_name: "SMILEtrack".to_string(),
            show_visualization: args.visualize,
            show_detections: false, // Don't show detections for videos by default
            tracking_log: Vec::new(),
            tracking_log_path,
        };
        
        // Process frames
        let mut frame = Mat::default();
        let mut frame_id = 0;
        
        while cap.read(&mut frame)? {
            if frame.empty() {
                break;
            }
            
            // Process frame
            if !processing_state.process_frame(&frame, None, frame_id, fps)? {
                // Processing was interrupted by user
                break;
            }
            
            frame_id += 1;
            
            // Print progress
            if frame_id % 10 == 0 {
                println!("Processed {}/{} frames", frame_id, total_frames);
            }
        }
        
        println!("\nVideo processing completed!");
        println!("Processed {} frames", frame_id);
    }
    
    // Get the tracking log path from the command line arguments instead of processing_state
    println!("Tracking completed successfully");
    if let Some(output_path) = &args.output {
        let log_path = if output_path.is_dir() {
            output_path.join("tracking_details.json")
        } else if let Some(parent) = output_path.parent() {
            parent.join("tracking_details.json")
        } else {
            PathBuf::from("tracking_details.json")
        };
        
        if log_path.exists() {
            println!("Tracking details saved to {:?}", log_path);
        }
    } else {
        println!("Tracking details saved to tracking_details.json");
    }
    
    Ok(())
} 