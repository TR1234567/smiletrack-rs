use anyhow::Result;
use opencv::{core::Size, imgcodecs, imgproc, prelude::*};
use std::path::Path;
use tch::{Device, Kind, Tensor};

// This is a minimal test program to print the raw output of the model
// It doesn't use any other code from the smiletrack crate so we can debug 
// issues with the model format directly
fn main() -> Result<()> {
    // Define model path
    let model_path = "weights/yolov7.torchscript";
    
    // Exit if model doesn't exist
    if !Path::new(model_path).exists() {
        println!("Model file not found at: {}", model_path);
        println!("Please ensure the model exists at this path");
        return Ok(());
    }
    
    println!("Loading model from: {}", model_path);
    
    // Load the model
    let model = tch::CModule::load(model_path)?;
    println!("Model loaded successfully");
    
    // Load test image
    let img_path = "image/test.jpg";
    println!("Loading test image from: {}", img_path);
    
    let img = imgcodecs::imread(img_path, imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        println!("Error: Could not load image from {}", img_path);
        return Ok(());
    }
    
    println!("Image loaded, size: {}x{}", img.cols(), img.rows());
    
    // Preprocess image
    println!("Preprocessing image...");
    let (input_w, input_h) = (640, 640);
    
    // Resize image
    let mut resized = Mat::default();
    imgproc::resize(&img, &mut resized, Size::new(input_w, input_h), 0.0, 0.0, imgproc::INTER_LINEAR)?;
    
    // Convert BGR to RGB
    let mut rgb = Mat::default();
    imgproc::cvt_color(&resized, &mut rgb, imgproc::COLOR_BGR2RGB, 0)?;
    
    // Convert to float32 and normalize
    let mut float_mat = Mat::default();
    rgb.convert_to(&mut float_mat, opencv::core::CV_32F, 1.0/255.0, 0.0)?;
    
    // Get dimensions and data
    let rows = float_mat.rows();
    let cols = float_mat.cols();
    let channels = float_mat.channels();
    let total_elements = (rows * cols * channels) as usize;
    let data = unsafe { std::slice::from_raw_parts(float_mat.data() as *const f32, total_elements) };
    
    // Create tensor with correct shape [1, C, H, W]
    let tensor = Tensor::from_slice(data)
        .reshape(&[1, channels as i64, rows as i64, cols as i64])
        .to_device(Device::Cpu)
        .to_kind(Kind::Float);
    
    println!("Input tensor shape: {:?}", tensor.size());
    
    // Run inference
    println!("Running model inference...");
    let output = model.forward_ts(&[&tensor])?;
    
    // Print output shape and type
    println!("Output tensor shape: {:?}", output.size());
    println!("Output tensor type: {:?}", output.kind());
    
    // Print sample values from the output
    println!("\nSample output values:");
    let output_shape = output.size();
    
    // Try to handle different output shapes
    if output_shape.len() >= 3 {
        if output_shape[2] == 6 {
            // Looks like [batch, detections, 6] format
            // Sample the first few detections
            let n_samples = std::cmp::min(5, output_shape[1] as usize);
            
            for i in 0..n_samples {
                let x = output.get(0).get(i as i64).get(0).double_value(&[]);
                let y = output.get(0).get(i as i64).get(1).double_value(&[]);
                let w = output.get(0).get(i as i64).get(2).double_value(&[]);
                let h = output.get(0).get(i as i64).get(3).double_value(&[]);
                let conf = output.get(0).get(i as i64).get(4).double_value(&[]);
                let cls = output.get(0).get(i as i64).get(5).double_value(&[]);
                
                println!("Detection {}: [x={:.3}, y={:.3}, w={:.3}, h={:.3}], conf={:.3}, class={}", 
                         i, x, y, w, h, conf, cls);
            }
        } else {
            // Might be raw YOLOv7 output [batch, anchors, values]
            // Sample from first few anchors
            let n_samples = std::cmp::min(5, output_shape[1] as usize);
            
            for i in 0..n_samples {
                print!("Box {}: [", i);
                for j in 0..std::cmp::min(6, output_shape[2] as usize) {
                    let value = output.get(0).get(i as i64).get(j as i64).double_value(&[]);
                    print!("{:.6}, ", value);
                }
                println!("]");
            }
        }
    } else {
        // Some other format
        println!("Unsupported output format, dimensions: {:?}", output_shape);
    }
    
    println!("\nDone!");
    Ok(())
} 