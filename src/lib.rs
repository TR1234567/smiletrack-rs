pub mod config;
pub mod detection;
pub mod utils;
pub mod tracker;
pub mod visualization;
pub mod simple_detector;

// Re-export main types
pub use crate::config::Config;
pub use crate::detection::{Detection, Detector};
pub use crate::tracker::{STrack, SMILEtrack};