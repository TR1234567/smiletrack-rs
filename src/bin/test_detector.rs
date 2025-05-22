use std::time::Instant;
use anyhow::Result;
use opencv::{imgcodecs, imgproc, prelude::*};
use smiletrack::{Detection, Config};

fn main() -> Result<()> {
    // Load config from file
    let mut config = Config::from_file("config.json")?;
    // Override some settings for testing
    config.model_path = "weights/yolov7.torchscript".to_string();
    config.conf_threshold = 0.001;
    config.nms_threshold = 0.45;
    
    println!("Testing detector with config: {:?}", config);
    
    println!("Using allowed classes: {:?}", config.classes);
    
    // Initialize detector
    let detector = smiletrack::detection::Detector::new(
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
    
    println!("Image loaded, size: {}x{}", img.cols(), img.rows());
    
    // Run detection
    let start = Instant::now();
    let detections = detector.detect(&img)?;
    let elapsed = start.elapsed();
    
    println!("Detection completed in {:?}", elapsed);
    println!("Found {} detections", detections.len());
    
    // Output detection details for debugging
    for (i, det) in detections.iter().enumerate() {
        println!("Detection {}: class={}, score={:.3}, box=[{:.1}, {:.1}, {:.1}, {:.1}]", 
                i, det.class_id, det.confidence, 
                det.tlwh[0], det.tlwh[1], det.tlwh[2], det.tlwh[3]);
    }
    
    // Create two visualizations to compare
    // 1. Original format
    let mut vis_img1 = img.clone();
    draw_detections(&mut vis_img1, &detections, false)?;
    let output_path1 = "detector_output_original.jpg";
    imgcodecs::imwrite(output_path1, &vis_img1, &opencv::core::Vector::new())?;
    
    // 2. Try with adjusted coordinates
    let mut vis_img2 = img.clone();
    draw_detections(&mut vis_img2, &detections, true)?;
    let output_path2 = "detector_output_adjusted.jpg";
    imgcodecs::imwrite(output_path2, &vis_img2, &opencv::core::Vector::new())?;
    
    println!("Visualizations saved to {} and {}", output_path1, output_path2);
    
    Ok(())
}

// Custom drawing function to test both formats
fn draw_detections(frame: &mut opencv::core::Mat, 
                 detections: &[Detection], 
                 adjust_format: bool) -> Result<()> {
    for det in detections {
        let tlwh = det.tlwh();
        let color = match det.class_id {
            0 => opencv::core::Scalar::new(0.0, 255.0, 0.0, 0.0), // Green for person
            2 => opencv::core::Scalar::new(255.0, 0.0, 0.0, 0.0), // Blue for car
            _ => opencv::core::Scalar::new(0.0, 0.0, 255.0, 0.0), // Red for others
        };
        
        // Draw boxes in two different ways to see which one works
        if adjust_format {
            // Interpret as center-format and convert to corner
            let cx = tlwh[0];
            let cy = tlwh[1];
            let w = tlwh[2];
            let h = tlwh[3];
            
            let x1 = cx - w/2.0;
            let y1 = cy - h/2.0;
            let x2 = cx + w/2.0;
            let y2 = cy + h/2.0;
            
            let rect = opencv::core::Rect::new(
                x1 as i32, y1 as i32,
                (x2-x1) as i32, (y2-y1) as i32
            );
            imgproc::rectangle(frame, rect, color, 2, imgproc::LINE_8, 0)?;
            
            // Draw with class and confidence
            let label = format!("{}:{:.2}", det.class_id, det.confidence);
            let text_pos = opencv::core::Point::new(x1 as i32, y1 as i32 - 5);
            imgproc::put_text(
                frame,
                &label,
                text_pos,
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                imgproc::LINE_8,
                false,
            )?;
        } else {
            // Use original format (just draw as is)
            let x = tlwh[0] as i32;
            let y = tlwh[1] as i32;
            let w = tlwh[2] as i32;
            let h = tlwh[3] as i32;
            
            let rect = opencv::core::Rect::new(x, y, w, h);
            imgproc::rectangle(frame, rect, color, 2, imgproc::LINE_8, 0)?;
            
            // Draw with class and confidence
            let label = format!("{}:{:.2}", det.class_id, det.confidence);
            let text_pos = opencv::core::Point::new(x, y - 5);
            imgproc::put_text(
                frame,
                &label,
                text_pos,
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                imgproc::LINE_8,
                false,
            )?;
        }
    }
    Ok(())
} 