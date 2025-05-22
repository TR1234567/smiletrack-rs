use serde::Deserialize;
use std::fs;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub model_path: String,
    pub track_high_thresh: f32,
    pub track_low_thresh: f32,
    pub new_track_thresh: f32,
    pub track_buffer: usize,
    pub proximity_thresh: f32,
    pub appearance_thresh: f32,
    pub with_reid: bool,
    pub device: String,
    pub input_size: [i32; 2],
    pub conf_threshold: f32,
    pub nms_threshold: f32,
    pub classes: Vec<i32>,
    // … other fields from config.json …
}

impl Config {
    /// Load from a JSON file.
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let data = fs::read_to_string(path)?;
        let cfg: Config = serde_json::from_str(&data)?;
        Ok(cfg)
    }
}