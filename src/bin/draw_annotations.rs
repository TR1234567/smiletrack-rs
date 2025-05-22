use clap::Parser;
use opencv::{
    core::{Point, Scalar, Rect},
    imgcodecs, imgproc,
    prelude::*,
};
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};

#[derive(Serialize, Deserialize, Debug, Clone)]
struct BoundingBox {
    class_id: i32,
    confidence: f32,
    x_center: f32,
    y_center: f32,
    width: f32,
    height: f32,
}

#[derive(Serialize, Deserialize, Debug)]
struct Annotation {
    frame: String,
    annotations: Vec<BoundingBox>,
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the annotations JSON file
    #[arg(short, long)]
    annotations: PathBuf,

    /// Path to the input image file
    #[arg(short, long)]
    input: PathBuf,

    /// Path to save the output image with bounding boxes
    #[arg(short, long)]
    output: PathBuf,

    /// Draw class ID and confidence
    #[arg(short, long)]
    show_labels: bool,
    
    /// Confidence threshold (0.0-1.0) to filter detections
    #[arg(short, long, default_value = "0.0")]
    threshold: f32,
    
    /// Maximum number of detections to show
    #[arg(short, long, default_value = "100")]
    max_detections: usize,
    
    /// Only show person detections (class_id = 0)
    #[arg(long)]
    only_persons: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Read annotations file
    println!("Reading annotations from {:?}", args.annotations);
    let json_content = fs::read_to_string(args.annotations)?;
    let annotations: Vec<Annotation> = serde_json::from_str(&json_content)?;

    // Find annotation for the input image
    let input_path = args.input.to_string_lossy().to_string();
    println!("Looking for annotations for: {}", input_path);
    
    // Create a more flexible matching solution - look for annotations with the same filename
    let input_filename = args.input.file_name().unwrap().to_string_lossy().to_string();
    
    let matching_annotations: Vec<&Annotation> = annotations.iter()
        .filter(|a| {
            let annotation_path = PathBuf::from(&a.frame);
            let annotation_filename = annotation_path.file_name().unwrap_or_default().to_string_lossy().to_string();
            annotation_filename == input_filename || a.frame.contains(&input_filename)
        })
        .collect();

    if matching_annotations.is_empty() {
        println!("No annotations found for {}", input_path);
        println!("Available frames in the annotation file:");
        for (i, ann) in annotations.iter().enumerate().take(5) {
            println!("  {}: {}", i, ann.frame);
        }
        if annotations.len() > 5 {
            println!("  ... and {} more", annotations.len() - 5);
        }
        return Err("No matching annotations found".into());
    }
    
    let annotation = &matching_annotations[0];
    println!("Found {} annotations for {}", annotation.annotations.len(), annotation.frame);

    // Read the input image
    println!("Reading image from {:?}", args.input);
    let mut img = imgcodecs::imread(&input_path, imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        return Err(format!("Failed to read image: {:?}", args.input).into());
    }
    
    let img_width = img.cols() as f32;
    let img_height = img.rows() as f32;
    
    // Define COCO class names (or relevant classes for your model)
    let class_names = vec![
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", 
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", 
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", 
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", 
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ];
    
    // Define colors for different classes
    let colors = vec![
        Scalar::new(0.0, 255.0, 0.0, 0.0),    // Green
        Scalar::new(255.0, 0.0, 0.0, 0.0),    // Blue
        Scalar::new(0.0, 0.0, 255.0, 0.0),    // Red
        Scalar::new(255.0, 255.0, 0.0, 0.0),  // Cyan
        Scalar::new(0.0, 255.0, 255.0, 0.0),  // Yellow
        Scalar::new(255.0, 0.0, 255.0, 0.0),  // Magenta
    ];

    // Filter and sort annotations by confidence
    let mut filtered_annotations: Vec<&BoundingBox> = annotation.annotations.iter()
        .filter(|bbox| bbox.confidence >= args.threshold)
        .filter(|bbox| !args.only_persons || bbox.class_id == 0)
        .collect();
    
    // Sort by confidence (highest first)
    filtered_annotations.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    
    // Limit to max_detections
    if filtered_annotations.len() > args.max_detections {
        filtered_annotations.truncate(args.max_detections);
    }
    
    println!("Drawing {} detections (filtered from {})",
             filtered_annotations.len(), annotation.annotations.len());

    // Count drawn boxes
    let mut drawn_count = 0;
    
    // Draw bounding boxes
    for bbox in &filtered_annotations {
        // Convert normalized coordinates to pixel coordinates
        let x_center = bbox.x_center * img_width;
        let y_center = bbox.y_center * img_height;
        let width = bbox.width * img_width;
        let height = bbox.height * img_height;
        
        // Calculate top-left and bottom-right corners
        let x1 = (x_center - width / 2.0) as i32;
        let y1 = (y_center - height / 2.0) as i32;
        let x2 = (x_center + width / 2.0) as i32;
        let y2 = (y_center + height / 2.0) as i32;
        
        // Ensure coordinates are within image bounds
        let x1 = x1.max(0).min(img_width as i32 - 1);
        let y1 = y1.max(0).min(img_height as i32 - 1);
        let x2 = x2.max(0).min(img_width as i32 - 1);
        let y2 = y2.max(0).min(img_height as i32 - 1);
        
        // Skip tiny boxes
        if x2 - x1 < 10 || y2 - y1 < 10 {
            continue;
        }
        
        drawn_count += 1;
        
        // Get color based on class_id
        let color = colors[bbox.class_id as usize % colors.len()];
        
        // Draw rectangle
        let rect = Rect::new(x1, y1, x2 - x1, y2 - y1);
        imgproc::rectangle(
            &mut img,
            rect,
            color,
            2, // Line thickness
            imgproc::LINE_8,
            0,
        )?;
        
        // Add label if requested
        if args.show_labels {
            let class_name = if bbox.class_id >= 0 && (bbox.class_id as usize) < class_names.len() {
                class_names[bbox.class_id as usize]
            } else {
                "unknown"
            };
            
            let text = format!("{} {:.2}", class_name, bbox.confidence);
            
            // Calculate text size
            let mut baseline = 0;
            let font_face = imgproc::FONT_HERSHEY_SIMPLEX;
            let font_scale = 0.5;
            let thickness = 1;
            
            let text_size = imgproc::get_text_size(
                &text,
                font_face,
                font_scale,
                thickness,
                &mut baseline,
            )?;
            
            // Draw filled rectangle for text background
            let bg_rect = Rect::new(
                x1, 
                y1 - text_size.height - 5,
                text_size.width,
                text_size.height + 5
            );
            
            imgproc::rectangle(
                &mut img,
                bg_rect,
                color,
                -1, // Fill rectangle
                imgproc::LINE_8,
                0,
            )?;
            
            // Draw text
            imgproc::put_text(
                &mut img,
                &text,
                Point::new(x1, y1 - 5),
                font_face,
                font_scale,
                Scalar::new(255.0, 255.0, 255.0, 0.0), // White text
                thickness,
                imgproc::LINE_8,
                false,
            )?;
        }
    }

    // Save output image
    println!("Saving output image to {:?}", args.output);
    imgcodecs::imwrite(
        &args.output.to_string_lossy(),
        &img,
        &opencv::core::Vector::new(),
    )?;
    
    println!("Done! Saved image with {} bounding boxes", drawn_count);
    
    Ok(())
} 