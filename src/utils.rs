/// Perform non-max suppression on boxes & scores, return indices to keep.
use opencv::{core::{Scalar, Point}, imgproc, prelude::*};
use nalgebra::{Matrix, Const, ArrayStorage};

pub fn nms(boxes: &[[f32; 4]], scores: &[f32], iou_thresh: f32) -> Vec<usize> {
    let mut idxs: Vec<usize> = (0..boxes.len()).collect();
    idxs.sort_unstable_by(|&i, &j| scores[j].partial_cmp(&scores[i]).unwrap());
    let mut keep = Vec::new();
    while let Some(&i) = idxs.first() {
        keep.push(i);
        idxs = idxs.into_iter()
            .skip(1)
            .filter(|&j| compute_iou_array(&boxes[i], &boxes[j]) < iou_thresh)
            .collect();
    }
    keep
}
pub fn draw_box(img: &mut Mat, bbox: [i32; 4], color: Scalar, thickness: i32) -> opencv::Result<()> {
    let rect = opencv::core::Rect::new(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);
    imgproc::rectangle(img, rect, color, thickness, imgproc::LINE_8, 0)
}
pub fn put_text(
    img: &mut Mat,
    text: &str,
    org: (i32, i32),
    color: Scalar,
    font_scale: f64,
    thickness: i32,
) -> opencv::Result<()> {
    let point = Point::new(org.0, org.1);
    imgproc::put_text(
        img,
        text,
        point,
        imgproc::FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        imgproc::LINE_8,
        false,
    )
}
// timing utility
pub fn now_ms() -> u128 {
    std::time::Instant::now().elapsed().as_millis()
}

/// Compute IoU between two bounding boxes as arrays: [x1, y1, w, h]
pub fn compute_iou_array(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let a_x1 = a[0];
    let a_y1 = a[1];
    let a_x2 = a[0] + a[2];
    let a_y2 = a[1] + a[3];
    
    let b_x1 = b[0];
    let b_y1 = b[1];
    let b_x2 = b[0] + b[2];
    let b_y2 = b[1] + b[3];
    
    compute_iou_tlbr(a_x1, a_y1, a_x2, a_y2, b_x1, b_y1, b_x2, b_y2)
}

/// Compute IoU between two bounding boxes as matrices
pub fn compute_iou(a: &Matrix<f32, Const<4>, Const<1>, ArrayStorage<f32, 4, 1>>, 
                  b: &Matrix<f32, Const<4>, Const<1>, ArrayStorage<f32, 4, 1>>) -> f32 {
    let a_x1 = a[0];
    let a_y1 = a[1];
    let a_x2 = a[0] + a[2];
    let a_y2 = a[1] + a[3];
    
    let b_x1 = b[0];
    let b_y1 = b[1];
    let b_x2 = b[0] + b[2];
    let b_y2 = b[1] + b[3];
    
    compute_iou_tlbr(a_x1, a_y1, a_x2, a_y2, b_x1, b_y1, b_x2, b_y2)
}

/// Helper function to compute IoU from top-left and bottom-right coordinates
fn compute_iou_tlbr(a_x1: f32, a_y1: f32, a_x2: f32, a_y2: f32, 
                   b_x1: f32, b_y1: f32, b_x2: f32, b_y2: f32) -> f32 {
    let x1 = a_x1.max(b_x1);
    let y1 = a_y1.max(b_y1);
    let x2 = a_x2.min(b_x2);
    let y2 = a_y2.min(b_y2);
    
    let inter_area = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let a_area = (a_x2 - a_x1) * (a_y2 - a_y1);
    let b_area = (b_x2 - b_x1) * (b_y2 - b_y1);
    
    if a_area + b_area - inter_area <= 0.0 {
        return 0.0;
    }
    
    inter_area / (a_area + b_area - inter_area)
}