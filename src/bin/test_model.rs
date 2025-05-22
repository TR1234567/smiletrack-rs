use anyhow::Result;
use opencv::{imgcodecs, prelude::*};
use tch::{Device, Kind, Tensor};

fn main() -> Result<()> {
    println!("Running model format test...");
    
    // Load model - adjust path as needed
    let model_path = "weights/yolov7.torchscript";
    println!("Loading model from: {}", model_path);
    let model = tch::CModule::load(model_path)?;
    
    // Load a test image
    let img_path = "image/test.jpg"; // Using the existing test.jpg file
    println!("Loading test image from: {}", img_path);
    let img = imgcodecs::imread(img_path, imgcodecs::IMREAD_COLOR)?;
    
    if img.empty() {
        println!("Error: Could not load image from {}", img_path);
        return Ok(());
    }
    
    println!("Image loaded, size: {}x{}", img.cols(), img.rows());
    
    // Preprocess image
    let mut resized = Mat::default();
    opencv::imgproc::resize(
        &img,
        &mut resized,
        opencv::core::Size::new(640, 640),
        0.0,
        0.0,
        opencv::imgproc::INTER_LINEAR,
    )?;
    
    // Convert BGR to RGB and normalize
    let mut rgb = Mat::default();
    opencv::imgproc::cvt_color(&resized, &mut rgb, opencv::imgproc::COLOR_BGR2RGB, 0)?;
    
    let mut float_mat = Mat::default();
    rgb.convert_to(&mut float_mat, opencv::core::CV_32F, 1.0/255.0, 0.0)?;
    
    // Convert to tensor
    let rows = float_mat.rows();
    let cols = float_mat.cols();
    let channels = float_mat.channels();
    let total_elements = (rows * cols * channels) as usize;
    let data = unsafe { std::slice::from_raw_parts(float_mat.data() as *const f32, total_elements) };
    
    // Create tensor with shape [1, 3, H, W]
    let tensor = Tensor::from_slice(data)
        .reshape(&[1, channels as i64, rows as i64, cols as i64])
        .to_device(Device::Cpu)
        .to_kind(Kind::Float);
    
    println!("Input tensor shape: {:?}", tensor.size());
    
    // Run inference
    println!("Running model inference...");
    let output = model.forward_ts(&[&tensor])?;
    
    // Print output tensor info
    println!("Output tensor shape: {:?}", output.size());
    println!("Output tensor type: {:?}", output.kind());
    
    // Check the first few values to understand format
    let batch_size = output.size()[0];
    
    println!("\nAnalyzing output tensor format...");
    
    if output.size().len() < 2 {
        println!("Unexpected output format: tensor has less than 2 dimensions");
        return Ok(());
    }
    
    // Try to determine if output is in [x,y,w,h] or [x1,y1,x2,y2] format
    // by analyzing the values
    println!("\nSample detections from first batch:");
    
    let num_detections = output.size()[1].min(5); // Show at most 5 detections
    
    for i in 0..num_detections {
        let box_data = output.get(0).get(i as i64);
        
        if box_data.size().len() < 2 {
            // Attempt to interpret as YOLOv7 output format: 
            // [x, y, w, h, obj_conf, class_scores...]
            let x = box_data.get(0).double_value(&[]);
            let y = box_data.get(1).double_value(&[]);
            let w = box_data.get(2).double_value(&[]);
            let h = box_data.get(3).double_value(&[]);
            let conf = box_data.get(4).double_value(&[]);
            
            println!("Detection {}: [x={:.3}, y={:.3}, w={:.3}, h={:.3}], conf={:.3}", 
                    i, x, y, w, h, conf);
            
            // Analyze if these are likely center coordinates or corner coordinates
            if w > 1.0 && h > 1.0 {
                println!("  Format appears to be [x1,y1,x2,y2] (corner coordinates)");
            } else if w < 1.0 && h < 1.0 {
                println!("  Format appears to be [cx,cy,w,h] (center coordinates, normalized)");
            } else {
                println!("  Format is unclear");
            }
        } else {
            // Handle other tensor formats
            println!("Detection {}: complex tensor format, shape: {:?}", i, box_data.size());
        }
    }
    
    // Save a simple visualization to verify model works at all
    let mut output_img = img.clone();
    // Just draw a test rectangle to make sure visualization works
    opencv::imgproc::rectangle(
        &mut output_img, 
        opencv::core::Rect::new(100, 100, 200, 200),
        opencv::core::Scalar::new(0.0, 255.0, 0.0, 0.0),
        2,
        opencv::imgproc::LINE_8,
        0
    )?;
    
    let output_path = "debug_output.jpg";
    imgcodecs::imwrite(output_path, &output_img, &opencv::core::Vector::new())?;
    println!("Debug image saved to {}", output_path);

    Ok(())
} 