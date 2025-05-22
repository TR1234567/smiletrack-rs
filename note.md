# Porting SMILEtrack to Rust

## Progress Checklist
- [x] Project Setup
  - [x] Create Rust crate structure
  - [x] Initial Cargo.toml with core dependencies
  - [x] Module organization (lib.rs)
- [x] Config Module (2.A)
  - [x] Config struct with serde
  - [x] JSON deserialization
- [x] Utils Module (2.C)
  - [x] IoU computation
  - [x] NMS implementation
  - [x] OpenCV drawing wrappers
- [x] Tracker Module (2.D)
  - [x] KalmanFilter base implementation
  - [x] Complete KalmanFilter update & gating
  - [x] STrack implementation
  - [x] Data association (Hungarian)
  - [x] GMC (optical flow)
    - [x] Feature point detection
    - [x] Optical flow estimation
    - [x] Homography computation
    - [x] Track motion compensation
  - [x] Unit tests
    - [x] KalmanFilter prediction/update
    - [x] STrack state transitions
    - [x] SMILEtrack matching
- [x] Detection Module (2.B)
  - [x] Model conversion (YOLOv7 → TorchScript)
  - [x] Frame preprocessing
  - [x] Inference pipeline
  - [x] Postprocessing & NMS
    - [x] YOLOv7 output decoding
    - [x] Confidence thresholding
    - [x] Coordinate conversion
    - [x] Non-maximum suppression
  - [x] Unit tests
    - [x] Preprocessing validation
    - [x] Inference validation
    - [x] End-to-end testing
- [ ] Application/CLI (2.E)
  - [ ] Command-line parsing
  - [ ] Video I/O pipeline
  - [ ] End-to-end integration
  - [ ] Visualization helpers

## Step 2A: Config
- Created `config::Config` struct mirroring Python `config.json`.
- Used `serde_json` for deserialization.

## Step 2B: Detection
- Completed YOLOv7 model conversion to TorchScript format.
- Implemented `Detector` with full inference pipeline:
  - Frame preprocessing (resize, normalize, tensor conversion)
  - Model inference with TorchScript
  - Output postprocessing:
    - Confidence thresholding
    - Class score computation
    - Coordinate conversion to original frame size
    - Non-maximum suppression
- Added comprehensive unit tests:
  - Model initialization
  - Preprocessing validation
  - Inference pipeline testing
  - End-to-end detection testing

## Step 2C: Utils
- Implemented `compute_iou` helper for box IoU.
- Completed greedy `nms` suppression in `utils.rs`.
- Added `draw_box` and `put_text` wrappers around OpenCV drawing.

## Step 2D: Tracker
- Completed `KalmanFilter` implementation:
  - `new()` to initialize motion & update matrices
  - `initiate()` to create initial mean & covariance
  - `predict()` for the motion‐model prediction step
  - `project()` for mapping state to measurement space
  - `update()` with numerically stable correction
  - `gating_distance()` for measurement validation
- Implemented full `STrack` with:
  - State management (New → Tracked → Lost → Removed)
  - Feature & class history smoothing
  - Coordinate conversion utilities
- Added `SMILEtrack` multi-object tracker:
  - Track lifecycle management
  - IoU + Hungarian matching
  - Duplicate removal
- Added `GMC` for motion compensation:
  - Feature detection with `goodFeaturesToTrack`
  - Optical flow with `calcOpticalFlowPyrLK`
  - RANSAC homography estimation
  - Track state transformation
- Added comprehensive unit tests

## Step 2E: Application/CLI
- Created basic CLI structure with clap
- Added main.rs with initial pipeline:
  - Config loading
  - Single image detection demo
  - Box visualization
- TODO: Add video pipeline and tracking integration

## Step 3: Library Integration Notes
- Using `tch` for TorchScript model loading
- Using `nalgebra` for Kalman filter matrix operations
- Using `opencv` for:
  - Frame I/O
  - Drawing utilities
  - Optical flow and homography
- Using `lap` crate for Hungarian matching (implemented)

## Step 4: Next Steps
1. Complete Application/CLI:
   - [ ] Add video I/O pipeline
   - [ ] Implement visualization helpers
   - [ ] Add performance benchmarking
2. Testing & Documentation:
   - [ ] Add end-to-end integration tests
   - [ ] Document API and examples
3. Performance Optimization:
   - [ ] Profile and optimize detection pipeline
   - [ ] Optimize tracking updates
   - [ ] GPU acceleration where possible