use std::fs::File;
use std::io::Write;
use std::time::Instant;
use anyhow::Result;
use opencv::{imgcodecs, prelude::*};
use smiletrack::{Config, detection::Detector};

fn main() -> Result<()> {
    // Load config from file
    let mut config = Config::from_file("config.json")?;
    // Override some settings for testing
    config.model_path = "weights/yolov7.torchscript".to_string();
    config.conf_threshold = 0.4; // Higher threshold for cleaner comparison
    
    println!("Testing detector with config: {:?}", config);
    
    // Initialize detector
    let detector = Detector::new(
        &config.model_path,
        &config.device,
        (config.input_size[0] as i64, config.input_size[1] as i64),
        config.conf_threshold,
        config.nms_threshold,
    )?;
    
    // Load test image
    let img_path = "image/test.jpg";
    println!("Loading test image from: {}", img_path);
    let img = imgcodecs::imread(img_path, imgcodecs::IMREAD_COLOR)?;
    
    if img.empty() {
        println!("Error: Could not load image from {}", img_path);
        return Ok(());
    }
    
    // Run detection
    let start = Instant::now();
    let detections = detector.detect(&img)?;
    let elapsed = start.elapsed();
    
    println!("Detection completed in {:?}", elapsed);
    println!("Found {} detections", detections.len());
    
    // Write results to file for comparison
    let mut file = File::create("rust_detections.txt")?;
    writeln!(file, "# Rust detections for {}", img_path)?;
    writeln!(file, "# Format: class_id, confidence, x, y, w, h")?;
    
    for det in &detections {
        let tlwh = det.tlwh();
        writeln!(file, "{}, {:.4}, {:.1}, {:.1}, {:.1}, {:.1}", 
                det.class_id, det.confidence(), 
                tlwh[0], tlwh[1], tlwh[2], tlwh[3])?;
    }
    
    println!("Detection results written to rust_detections.txt");
    
    // Create visualization
    let mut vis_img = img.clone();
    for det in &detections {
        let tlwh = det.tlwh();
        let color = match det.class_id {
            0 => opencv::core::Scalar::new(0.0, 255.0, 0.0, 0.0), // Green for person
            2 => opencv::core::Scalar::new(255.0, 0.0, 0.0, 0.0), // Blue for car
            _ => opencv::core::Scalar::new(0.0, 0.0, 255.0, 0.0), // Red for others
        };
        
        let rect = opencv::core::Rect::new(
            tlwh[0] as i32, tlwh[1] as i32, 
            tlwh[2] as i32, tlwh[3] as i32
        );
        
        opencv::imgproc::rectangle(
            &mut vis_img, 
            rect,
            color,
            2,
            opencv::imgproc::LINE_8,
            0
        )?;
        
        // Add class and confidence text
        let label = format!("{}:{:.2}", det.class_id, det.confidence());
        let text_pos = opencv::core::Point::new(tlwh[0] as i32, tlwh[1] as i32 - 5);
        opencv::imgproc::put_text(
            &mut vis_img,
            &label,
            text_pos,
            opencv::imgproc::FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            opencv::imgproc::LINE_8,
            false,
        )?;
    }
    
    // Save visualization
    let output_path = "rust_visualization.jpg";
    imgcodecs::imwrite(output_path, &vis_img, &opencv::core::Vector::new())?;
    println!("Visualization saved to {}", output_path);
    
    // Create a simple Python script to compare with
    let mut py_script = File::create("compare_detections.py")?;
    write!(py_script, r###"
import cv2
import numpy as np
import torch
import sys
import os

# Adjust these paths to match your environment
MODEL_PATH = "weights/your_yolov7_model.pt"  # Your original PyTorch model
IMG_PATH = "image/test_image.jpg"
CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.45

# Load model (use appropriate loading code for your model)
# This is just an example, adjust based on your Python implementation
try:
    sys.path.insert(0, '.')
    from models.experimental import attempt_load
    from utils.general import non_max_suppression, scale_coords
    
    device = torch.device('cpu')
    model = attempt_load(MODEL_PATH, map_location=device)
    model.eval()
    print(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Skipping Python detection")
    sys.exit(1)

# Load image
img = cv2.imread(IMG_PATH)
if img is None:
    print(f"Error: Could not load image from {IMG_PATH}")
    sys.exit(1)

# Prepare image
img_size = 640
img_tensor = cv2.resize(img, (img_size, img_size))
img_tensor = img_tensor.transpose(2, 0, 1)  # HWC to CHW
img_tensor = torch.from_numpy(img_tensor).float().div(255.0).unsqueeze(0)

# Run inference
with torch.no_grad():
    pred = model(img_tensor)[0]
    pred = non_max_suppression(pred, CONF_THRESHOLD, NMS_THRESHOLD)

# Process and save detections
with open("python_detections.txt", "w") as f:
    f.write(f"# Python detections for {IMG_PATH}\n")
    f.write("# Format: class_id, confidence, x, y, w, h\n")
    
    for i, det in enumerate(pred):
        if len(det):
            # Rescale boxes from img_size to original size
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img.shape).round()
            
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = [x.item() for x in xyxy]
                cls_id = int(cls.item())
                conf_val = conf.item()
                
                # Convert from xyxy (corner format) to xywh (width/height format)
                x, y = x1, y1
                w, h = x2 - x1, y2 - y1
                
                f.write(f"{cls_id}, {conf_val:.4f}, {x}, {y}, {w}, {h}\n")
                
                # Draw on image
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"{cls_id}:{conf_val:.2f}"
                cv2.putText(img, label, (int(x1), int(y1)-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save visualization
cv2.imwrite("python_visualization.jpg", img)
print("Python detection results saved to python_detections.txt")
print("Python visualization saved to python_visualization.jpg")

# Compare results
try:
    with open("rust_detections.txt", "r") as f:
        rust_lines = [line for line in f.readlines() if not line.startswith("#")]
    
    with open("python_detections.txt", "r") as f:
        python_lines = [line for line in f.readlines() if not line.startswith("#")]
    
    print(f"\nRust detections: {len(rust_lines)}")
    print(f"Python detections: {len(python_lines)}")
    
    print("\nComparing detection formats...")
    if len(rust_lines) > 0 and len(python_lines) > 0:
        # Parse a sample detection from each
        rust_sample = rust_lines[0].strip().split(", ")
        python_sample = python_lines[0].strip().split(", ")
        
        print(f"Rust format sample: {rust_sample}")
        print(f"Python format sample: {python_sample}")
    
    print("\nDone!")
except Exception as e:
    print(f"Error comparing results: {e}")
"###)?;
    
    println!("Created Python comparison script: compare_detections.py");
    println!("You can run it to compare with Python detections");
    
    Ok(())
} 