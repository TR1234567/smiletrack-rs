use opencv::{
    core::{Point, Scalar, Rect},
    imgproc,
    prelude::*,
};
use crate::{Detection, STrack};

#[allow(dead_code)]
const COLORS: &[Scalar] = &[
    Scalar::new(255.0, 0.0, 0.0, 0.0),    // Red
    Scalar::new(0.0, 255.0, 0.0, 0.0),    // Green
    Scalar::new(0.0, 0.0, 255.0, 0.0),    // Blue
    Scalar::new(255.0, 255.0, 0.0, 0.0),  // Yellow
    Scalar::new(255.0, 0.0, 255.0, 0.0),  // Magenta
    Scalar::new(0.0, 255.0, 255.0, 0.0),  // Cyan
];

/// Draw text on an image with specified font size and color
pub fn draw_text(
    frame: &mut Mat, 
    text: &str, 
    x: i32, 
    y: i32, 
    font_scale: f64, 
    color: (i32, i32, i32)
) -> opencv::Result<()> {
    let color = Scalar::new(color.2 as f64, color.1 as f64, color.0 as f64, 0.0); // BGR format
    let text_pos = Point::new(x, y);
    
    // Add a black outline for better visibility
    imgproc::put_text(
        frame,
        text,
        text_pos,
        imgproc::FONT_HERSHEY_SIMPLEX,
        font_scale,
        Scalar::new(0.0, 0.0, 0.0, 0.0), // Black
        3, // Thicker line for background
        imgproc::LINE_8,
        false,
    )?;
    
    // Add the actual text on top
    imgproc::put_text(
        frame,
        text,
        text_pos,
        imgproc::FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        1,
        imgproc::LINE_8,
        false,
    )?;
    
    Ok(())
}

pub fn draw_track(frame: &mut Mat, track: &STrack, color: Scalar) -> anyhow::Result<()> {
    if !track.is_activated() {
        return Ok(());
    }

    let tlwh = track.tlwh();
    let track_id = track.track_id();
    
    let tl = Point::new(tlwh[0] as i32, tlwh[1] as i32);
    let br = Point::new((tlwh[0] + tlwh[2]) as i32, (tlwh[1] + tlwh[3]) as i32);
    
    let rect = Rect::new(tl.x, tl.y, br.x - tl.x, br.y - tl.y);
    imgproc::rectangle(frame, rect, color, 2, imgproc::LINE_8, 0)?;
    
    let text = format!("ID: {}", track_id);
    let mut baseline = 0;
    let _text_size = imgproc::get_text_size(&text, imgproc::FONT_HERSHEY_SIMPLEX, 0.5, 1, &mut baseline)?;
    let text_org = Point::new(tl.x, tl.y - 5);
    imgproc::put_text(
        frame,
        &text,
        text_org,
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        imgproc::LINE_8,
        false,
    )?;

    if let Some(trail) = track.motion_trail() {
        for i in 1..trail.len() {
            let prev = &trail[i-1];
            let curr = &trail[i];
            let prev_pt = Point::new(prev[0] as i32, prev[1] as i32);
            let curr_pt = Point::new(curr[0] as i32, curr[1] as i32);
            imgproc::line(frame, prev_pt, curr_pt, color, 1, imgproc::LINE_8, 0)?;
        }
    }

    Ok(())
}

pub fn draw_detection(frame: &mut Mat, det: &Detection, color: Scalar) -> anyhow::Result<()> {
    let tlwh = det.tlwh();
    let score = det.confidence();
    
    let tl = Point::new(tlwh[0] as i32, tlwh[1] as i32);
    let br = Point::new((tlwh[0] + tlwh[2]) as i32, (tlwh[1] + tlwh[3]) as i32);
    
    let rect = Rect::new(tl.x, tl.y, br.x - tl.x, br.y - tl.y);
    imgproc::rectangle(frame, rect, color, 2, imgproc::LINE_8, 0)?;
    
    // Get class name based on class_id
    let display_name = match det.class_id {
        0 => "person".to_string(),
        1 => "bicycle".to_string(),
        2 => "car".to_string(),
        3 => "motorcycle".to_string(),
        5 => "bus".to_string(),
        7 => "truck".to_string(),
        9 => "traffic light".to_string(),
        15 => "cat".to_string(),
        16 => "dog".to_string(),
        _ => format!("object_{}", det.class_id),
    };
    
    // Format text with class name and confidence
    let text = format!("{} {:.2}", display_name, score);
    
    // Add text with dark background for better visibility
    let mut baseline = 0;
    let text_size = imgproc::get_text_size(&text, imgproc::FONT_HERSHEY_SIMPLEX, 0.5, 1, &mut baseline)?;
    
    // Draw background rectangle for text
    let bg_rect = Rect::new(
        tl.x, 
        tl.y - text_size.height - 5, 
        text_size.width, 
        text_size.height + 5
    );
    
    // Fill with semi-transparent background
    imgproc::rectangle(
        frame,
        bg_rect,
        Scalar::new(0.0, 0.0, 0.0, 0.0), // Black background
        -1, // Filled rectangle
        imgproc::LINE_8,
        0
    )?;
    
    // Draw text
    let text_org = Point::new(tl.x, tl.y - 5);
    imgproc::put_text(
        frame,
        &text,
        text_org,
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        imgproc::LINE_8,
        false,
    )?;

    Ok(())
}

pub fn draw_frame_info(frame: &mut Mat, frame_id: i32, fps: f64) -> opencv::Result<()> {
    let text = format!("Frame: {} FPS: {:.1}", frame_id, fps);
    let text_pos = Point::new(10, 30);
    opencv::imgproc::put_text(
        frame,
        &text,
        text_pos,
        opencv::imgproc::FONT_HERSHEY_SIMPLEX,
        0.6,
        Scalar::new(255.0, 255.0, 255.0, 0.0),
        2,
        opencv::imgproc::LINE_8,
        false,
    )?;
    Ok(())
}

/// Draw detections with a limit on how many to show
pub fn draw_detections(frame: &mut Mat, detections: &[Detection]) -> anyhow::Result<()> {
    // Limit the number of visualized detections to avoid cluttering
    const MAX_VISUALIZED_DETECTIONS: usize = 20;
    
    // Sort detections by confidence (highest first)
    let mut sorted_dets: Vec<&Detection> = detections.iter().collect();
    sorted_dets.sort_by(|a, b| b.confidence().partial_cmp(&a.confidence()).unwrap());
    
    // Only visualize the top N detections
    let vis_dets = if sorted_dets.len() > MAX_VISUALIZED_DETECTIONS {
        &sorted_dets[0..MAX_VISUALIZED_DETECTIONS]
    } else {
        &sorted_dets
    };
    
    for (i, det) in vis_dets.iter().enumerate() {
        let color = COLORS[i % COLORS.len()];
        draw_detection(frame, det, color)?;
    }
    
    Ok(())
}

/// Draw tracks with a limit on how many to show
pub fn draw_tracks(frame: &mut Mat, tracks: &[STrack]) -> anyhow::Result<()> {
    // Limit the number of visualized tracks to avoid cluttering
    const MAX_VISUALIZED_TRACKS: usize = 50;
    
    // Only visualize active tracks, up to the maximum
    let active_tracks: Vec<&STrack> = tracks.iter()
        .filter(|t| t.is_activated())
        .collect();
    
    let vis_tracks = if active_tracks.len() > MAX_VISUALIZED_TRACKS {
        &active_tracks[0..MAX_VISUALIZED_TRACKS]
    } else {
        &active_tracks
    };
    
    for track in vis_tracks {
        let color = COLORS[(track.track_id() as usize) % COLORS.len()];
        draw_track(frame, track, color)?;
    }
    
    Ok(())
} 