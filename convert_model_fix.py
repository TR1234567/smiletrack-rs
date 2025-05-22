#!/usr/bin/env python3
import torch
import argparse
import os
import sys

class ModelWrapper(torch.nn.Module):
    """
    A wrapper class to match the output format expected by the Rust code.
    This wrapper post-processes the raw YOLO output to get it in the correct format.
    """
    def __init__(self, model, conf_thresh=0.25):
        super().__init__()
        self.model = model
        self.conf_thresh = conf_thresh
    
    def forward(self, x):
        # Original model prediction
        raw_output = self.model(x)
        
        # Process the model output based on the model type
        if isinstance(raw_output, tuple):
            # Some models return a tuple of tensors
            if len(raw_output) == 2:
                # Model likely returns both raw output and processed boxes
                detections = raw_output[1]
            else:
                # Pick the first one and process
                detections = self._process_raw_output(raw_output[0])
        elif isinstance(raw_output, list):
            # Some models return a list of tensors
            detections = self._process_raw_output(raw_output[0])
        else:
            # Single tensor output
            detections = self._process_raw_output(raw_output)
        
        # Log the shape for debugging
        if detections.numel() > 0:
            print(f"Processed output shape: {detections.shape}")
        
        return detections
    
    def _process_raw_output(self, output):
        """Process the raw YOLO output to get boxes in [x1, y1, x2, y2, conf, class_id] format."""
        
        # Get output dimensions
        batch_size = output.shape[0]
        boxes_detected = []
        
        try:
            # For YOLOv7/v8-like output processing
            from prb.utils.general import non_max_suppression
            
            # Run NMS to get boxes in xyxy format with confidence and class
            detections = non_max_suppression(output, conf_thres=self.conf_thresh, iou_thres=0.45)
            
            # Stack results from all batches
            for i, det in enumerate(detections):
                if len(det):
                    boxes_detected.append(det)
            
            if boxes_detected:
                return torch.cat(boxes_detected, dim=0)
            else:
                # Return empty tensor with correct format
                return torch.zeros((0, 6), device=output.device)
                
        except (ImportError, NameError) as e:
            print(f"Could not use YOLOv7 processing: {e}")
            print("Falling back to manual processing")
            
            # Manual processing as fallback
            # This assumes raw output in format [batch, anchors, 5+classes]
            # where 5 = [x, y, w, h, obj_conf]
            
            # Extract coordinates and confidence
            box_xy = output[..., 0:2]
            box_wh = output[..., 2:4]
            box_conf = output[..., 4:5]
            box_cls = output[..., 5:]
            
            # Convert to corner format
            xmin = box_xy[..., 0:1] - box_wh[..., 0:1] / 2
            ymin = box_xy[..., 1:2] - box_wh[..., 1:2] / 2
            xmax = box_xy[..., 0:1] + box_wh[..., 0:1] / 2
            ymax = box_xy[..., 1:2] + box_wh[..., 1:2] / 2
            
            # Get class scores and IDs
            max_scores, max_cls_indices = torch.max(box_cls, dim=2)
            
            # Combine scores with object confidence
            scores = box_conf.squeeze(-1) * max_scores
            
            # Filter by confidence threshold
            mask = scores > self.conf_thresh
            
            # Process each batch
            for i in range(batch_size):
                batch_mask = mask[i]
                if not batch_mask.any():
                    continue
                
                # Extract filtered detections
                batch_boxes = torch.cat([
                    xmin[i, batch_mask],
                    ymin[i, batch_mask],
                    xmax[i, batch_mask],
                    ymax[i, batch_mask],
                    scores[i, batch_mask].unsqueeze(-1),
                    max_cls_indices[i, batch_mask].float().unsqueeze(-1)
                ], dim=1)
                
                boxes_detected.append(batch_boxes)
            
            if boxes_detected:
                return torch.cat(boxes_detected, dim=0)
            else:
                # Return empty tensor with correct format
                return torch.zeros((0, 6), device=output.device)


def convert_model(model_path, output_path, conf_thresh=0.25):
    """
    Convert a PyTorch model to TorchScript format with the correct output format.
    
    Args:
        model_path: Path to the .pt model file
        output_path: Path for saving the TorchScript model
        conf_thresh: Confidence threshold for detections
    """
    print(f"Loading model from {model_path}")
    
    # Attempt to add YOLOv7 code directories to the path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prb_dir = os.path.join(repo_root, "prb")
    
    # Add both the model directory and the PRB directory to path for imports
    model_dir = os.path.dirname(os.path.abspath(model_path))
    sys.path.insert(0, model_dir)
    sys.path.insert(0, prb_dir)
    sys.path.insert(0, repo_root)
    
    print(f"Added to Python path: {repo_root}")
    print(f"Added to Python path: {prb_dir}")
    print(f"Added to Python path: {model_dir}")
    
    try:
        # Try YOLOv7 loading using the prb code
        print("Trying to load with YOLOv7 attempt_load...")
        from prb.models.experimental import attempt_load
        model = attempt_load(model_path, map_location='cpu')
        print("Loaded model using YOLOv7 attempt_load")
    except (ImportError, NameError):
        try:
            # Try direct PyTorch loading
            model = torch.load(model_path, map_location='cpu')
            if isinstance(model, dict):
                # YOLO models are often saved as a dict with 'model' key
                if 'model' in model:
                    model = model['model']
                # Sometimes the model is in 'state_dict' format
                elif 'state_dict' in model:
                    print("Cannot load state_dict directly, need model definition")
                    return False
            print("Loaded model using torch.load")
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    # Create wrapped model that will produce the desired output format
    wrapped_model = ModelWrapper(model, conf_thresh=conf_thresh)
    
    # Set model to evaluation mode
    wrapped_model.eval()
    
    # Create a dummy input
    dummy_input = torch.zeros((1, 3, 640, 640), dtype=torch.float32)
    
    # Test the wrapped model
    with torch.no_grad():
        try:
            output = wrapped_model(dummy_input)
            print(f"Test run successful, output shape: {output.shape}")
        except Exception as e:
            print(f"Error during test run: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Convert to TorchScript
    try:
        print("Converting to TorchScript...")
        script_model = torch.jit.trace(wrapped_model, dummy_input)
        script_model.save(output_path)
        print(f"Model saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch YOLO model to TorchScript with proper output format")
    parser.add_argument("--input", type=str, required=True, help="Path to .pt model file")
    parser.add_argument("--output", type=str, required=True, help="Path for saving TorchScript model")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    
    args = parser.parse_args()
    
    # Convert the model
    success = convert_model(args.input, args.output, args.conf)
    
    if success:
        print("Conversion completed successfully!")
    else:
        print("Conversion failed.")
        sys.exit(1) 