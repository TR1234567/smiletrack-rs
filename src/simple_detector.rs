use anyhow::Result;
use opencv::{
    core::{Mat, Size, CV_32F},
    imgproc,
    prelude::*,
};
use serde::{Serialize, Deserialize};
use tch::{Device, Kind, Tensor};
use std::collections::HashMap;

/// Simple detection result structure that matches Python output format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleDetection {
    pub bbox: [f32; 4],      // [x, y, width, height]
    pub confidence: f32,
    pub class_id: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub class_name: Option<String>,
}

/// Simple frame result structure that matches Python output format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleFrameResult {
    pub frame_id: i32,
    pub detections: Vec<SimpleDetection>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub tracks: Vec<SimpleTrack>,
}

/// Simple track result structure that matches Python output format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleTrack {
    pub track_id: i32,
    pub bbox: [f32; 4],
    pub confidence: f32,
    pub class_id: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub class_name: Option<String>,
}

/// Simple detector that focuses only on producing detection outputs similar to Python
pub struct SimpleDetector {
    model: tch::CModule,
    device: Device,
    input_size: (i64, i64),
    pub conf_threshold: f32,
    pub nms_threshold: f32,
    pub allowed_classes: Vec<i32>,
    pub class_names: HashMap<i32, String>,
}

impl SimpleDetector {
    /// Create a new simple detector
    pub fn new(
        model_path: &str,
        device_str: &str,
        input_size: (i64, i64),
        conf_threshold: f32,
        nms_threshold: f32,
    ) -> Result<Self> {
        // Set device
        let device = if device_str == "cuda" && tch::Cuda::is_available() {
            Device::Cuda(0)
        } else {
            Device::Cpu
        };
        
        // Load model
        let model = tch::CModule::load(model_path)?;
        
        // Define default class names for COCO dataset
        let mut class_names = HashMap::new();
        class_names.insert(0, "person".to_string());
        class_names.insert(1, "bicycle".to_string());
        class_names.insert(2, "car".to_string());
        class_names.insert(3, "motorcycle".to_string());
        class_names.insert(5, "bus".to_string());
        class_names.insert(7, "truck".to_string());
        class_names.insert(15, "cat".to_string());
        class_names.insert(16, "dog".to_string());
        
        Ok(SimpleDetector {
            model,
            device,
            input_size,
            conf_threshold,
            nms_threshold,
            allowed_classes: vec![0, 1, 2, 3, 5, 7, 15, 16],
            class_names,
        })
    }
    
    /// Set allowed classes
    pub fn set_allowed_classes(&mut self, classes: Vec<i32>) {
        self.allowed_classes = classes;
    }
    
    /// Process a frame and return detections
    pub fn process_frame(&self, frame: &Mat, frame_id: i32) -> Result<SimpleFrameResult> {
        // Preprocess the frame
        let input_tensor = self.preprocess(frame)?;
        
        // Run inference
        let output = self.model.forward_ts(&[&input_tensor])?;
        
        // Post-process to get detections
        let detections = self.postprocess(&output, frame)?;
        
        // Create frame result
        let frame_result = SimpleFrameResult {
            frame_id,
            detections,
            tracks: Vec::new(),
        };
        
        Ok(frame_result)
    }
    
    /// Preprocess a frame for inference
    fn preprocess(&self, frame: &Mat) -> Result<Tensor> {
        // Get frame dimensions
        let orig_height = frame.rows() as f32;
        let orig_width = frame.cols() as f32;
        
        // Resize frame to input size
        let mut resized = Mat::default();
        imgproc::resize(
            frame,
            &mut resized,
            Size::new(self.input_size.0 as i32, self.input_size.1 as i32),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;
        
        // Convert BGR to RGB
        let mut rgb = Mat::default();
        imgproc::cvt_color(&resized, &mut rgb, imgproc::COLOR_BGR2RGB, 0)?;
        
        // Convert to float32 and normalize to [0,1]
        let mut float_mat = Mat::default();
        rgb.convert_to(&mut float_mat, CV_32F, 1.0/255.0, 0.0)?;
        
        // Convert to tensor
        let rows = float_mat.rows();
        let cols = float_mat.cols();
        let channels = float_mat.channels();
        let total_elements = (rows * cols * channels) as usize;
        
        let data = unsafe {
            std::slice::from_raw_parts(float_mat.data() as *const f32, total_elements)
        };
        
        // Create tensor with shape [1, C, H, W]
        let tensor = Tensor::from_slice(data)
            .reshape(&[1, channels as i64, rows as i64, cols as i64])
            .to_device(self.device)
            .to_kind(Kind::Float);
        
        Ok(tensor)
    }
    
    /// Post-process model output to get detections
    fn postprocess(&self, output: &Tensor, frame: &Mat) -> Result<Vec<SimpleDetection>> {
        // Get frame dimensions for scaling
        let frame_height = frame.rows() as f32;
        let frame_width = frame.cols() as f32;
        
        // Model input dimensions (assuming 640x640, common for YOLOv7)
        let model_input_width = self.input_size.0 as f32;
        let model_input_height = self.input_size.1 as f32;

        // Copy to CPU for processing
        let cpu_output = output.to_device(Device::Cpu);
        
        println!("Output tensor shape: {:?}", cpu_output.size());
        
        let mut final_detections = Vec::new(); // Renamed from detections to avoid confusion
        
        // Handle YOLOv7 output format [1, 25200, 85]
        if cpu_output.size().len() == 3 && cpu_output.size()[2] == 85 {
            let num_potential_boxes = cpu_output.size()[1];
            
            println!("Processing {} potential boxes from YOLOv7 output", num_potential_boxes);
            
            // Store intermediate detections: (x1, y1, x2, y2, obj_conf, class_id)
            // Coordinates are relative to model input size (e.g., 640x640)
            let mut pre_nms_detections: Vec<(f32, f32, f32, f32, f32, i32)> = Vec::new();

            // For debugging: print top raw objectness scores
            let mut raw_scores_for_debug = Vec::new();
            for i in 0..num_potential_boxes {
                raw_scores_for_debug.push(cpu_output.get(0).get(i).get(4).double_value(&[]) as f32);
            }
            raw_scores_for_debug.sort_by(|a, b| b.partial_cmp(a).unwrap());
            println!("Top 10 RAW objectness scores from tensor: {:?}", raw_scores_for_debug.iter().take(10).collect::<Vec<_>>());

            for i in 0..num_potential_boxes {
                let obj_conf_raw = cpu_output.get(0).get(i).get(4).double_value(&[]) as f32;
                let obj_conf_prob = 1.0 / (1.0 + (-obj_conf_raw).exp()); // Apply sigmoid to objectness score

                let mut max_cls_prob = 0.0f32;
                let mut class_id_for_this_box = 0i32;
                for c in 0..80 { // Assuming 80 classes
                    let cls_logit = cpu_output.get(0).get(i).get(5 + c).double_value(&[]) as f32;
                    let cls_prob = 1.0 / (1.0 + (-cls_logit).exp()); // Sigmoid on class score
                    if cls_prob > max_cls_prob {
                        max_cls_prob = cls_prob;
                        class_id_for_this_box = c as i32;
                    }
                }

                // Debug print for the first 10 boxes and any box where objectness_prob is somewhat high (e.g. > 0.1 after sigmoid)
                if i < 10 || obj_conf_prob > 0.1 {
                     println!(
                        "Debug Box Idx {}: raw_obj={:.4}, sig_obj={:.4}, cls_id={}, max_cls_prob={:.4}, combined_prob={:.4}",
                        i, obj_conf_raw, obj_conf_prob, class_id_for_this_box, max_cls_prob, obj_conf_prob * max_cls_prob
                    );
                }

                // Filter by objectness confidence
                if obj_conf_prob < self.conf_threshold {
                    continue;
                }
                
                // Filter by allowed classes
                if !self.allowed_classes.contains(&class_id_for_this_box) {
                    continue;
                }

                // Raw coordinates from model output (center_x, center_y, width, height)
                let cx = cpu_output.get(0).get(i).get(0).double_value(&[]) as f32;
                let cy = cpu_output.get(0).get(i).get(1).double_value(&[]) as f32;
                let w = cpu_output.get(0).get(i).get(2).double_value(&[]) as f32;
                let h = cpu_output.get(0).get(i).get(3).double_value(&[]) as f32;

                let x1 = cx - w / 2.0;
                let y1 = cy - h / 2.0;
                let x2 = cx + w / 2.0;
                let y2 = cy + h / 2.0;

                pre_nms_detections.push((x1, y1, x2, y2, obj_conf_prob, class_id_for_this_box));
            }
            
            println!("Found {} detections after initial confidence and class filtering (before NMS)", pre_nms_detections.len());

            // Sort by objectness confidence (descending) for NMS
            pre_nms_detections.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap());

            // Apply NMS
            let mut nms_selected_indices = Vec::new();
            let mut used_indices = vec![false; pre_nms_detections.len()];

            for i in 0..pre_nms_detections.len() {
                if used_indices[i] {
                    continue;
                }
                nms_selected_indices.push(i);
                used_indices[i] = true; // Mark as used

                let (x1_i, y1_i, x2_i, y2_i, _, cls_i) = pre_nms_detections[i];
                let area_i = (x2_i - x1_i).max(0.0) * (y2_i - y1_i).max(0.0);

                for j in (i + 1)..pre_nms_detections.len() {
                    if used_indices[j] {
                        continue;
                    }
                    let (x1_j, y1_j, x2_j, y2_j, _, cls_j) = pre_nms_detections[j];

                    if cls_i != cls_j {
                        continue;
                    }

                    let inter_x1 = x1_i.max(x1_j);
                    let inter_y1 = y1_i.max(y1_j);
                    let inter_x2 = x2_i.min(x2_j);
                    let inter_y2 = y2_i.min(y2_j);

                    let inter_w = (inter_x2 - inter_x1).max(0.0);
                    let inter_h = (inter_y2 - inter_y1).max(0.0);
                    let inter_area = inter_w * inter_h;
                    
                    let area_j = (x2_j - x1_j).max(0.0) * (y2_j - y1_j).max(0.0);
                    let union_area = area_i + area_j - inter_area;

                    if union_area > 0.0 {
                        let iou = inter_area / union_area;
                        if iou > self.nms_threshold { 
                            used_indices[j] = true;
                        }
                    }
                }
            }
            
            println!("Kept {} detections after NMS", nms_selected_indices.len());

            for &idx in &nms_selected_indices {
                let (x1_model, y1_model, x2_model, y2_model, obj_conf_prob, class_id) = pre_nms_detections[idx];

                let scale_w = frame_width / model_input_width;
                let scale_h = frame_height / model_input_height;

                let final_x1 = x1_model * scale_w;
                let final_y1 = y1_model * scale_h;
                let final_x2 = x2_model * scale_w;
                let final_y2 = y2_model * scale_h;
                
                let final_w = (final_x2 - final_x1).max(0.0);
                let final_h = (final_y2 - final_y1).max(0.0);

                if final_w * final_h < 10.0 { 
                    continue;
                }
                if final_w <= 0.0 || final_h <= 0.0 {
                    continue;
                }

                let class_name = self.class_names.get(&class_id).cloned();
                
                final_detections.push(SimpleDetection {
                    bbox: [final_x1, final_y1, final_w, final_h], 
                    confidence: obj_conf_prob, 
                    class_id,
                    class_name,
                });
            }
        } else {
            println!("Unexpected output tensor shape: {:?}", cpu_output.size());
        }
        
        println!("Found {} high confidence detections after all processing", final_detections.iter().filter(|d| d.confidence > 0.5).count());
        println!("Returning {} final detections", final_detections.len());
        Ok(final_detections)
    }
}