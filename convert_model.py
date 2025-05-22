import torch
import torch.serialization
import sys
import os

# Add YOLOv7 directory to Python path
yolov7_path = "/Users/witsarut/Downloads/Tracking/SMILEtrack/Citysurvey_api/yolov7"
if os.path.exists(yolov7_path):
    sys.path.insert(0, yolov7_path)
else:
    print(f"Error: YOLOv7 directory not found at {yolov7_path}")
    sys.exit(1)

# Import YOLOv7 modules
try:
    from models.yolo import Model
    from models.experimental import attempt_load
    from utils.general import non_max_suppression
    print("Successfully imported YOLOv7 modules")
except ImportError as e:
    print(f"Error importing YOLOv7 modules: {e}")
    print("Make sure the YOLOv7 repository is correctly installed")
    sys.exit(1)

def convert_pt_to_torchscript(pt_path, output_path):
    print(f"Loading model from {pt_path}")
    try:
        # Load the YOLOv7 model using the correct approach
        with torch.serialization.safe_globals(['models.yolo.Model']):
            # First try with attempt_load (preferred YOLOv7 way)
            try:
                model = attempt_load(pt_path, map_location=torch.device('cpu'))
                print("Model loaded using attempt_load")
            except Exception as e:
                print(f"attempt_load failed, trying direct load: {e}")
                # Fallback to direct loading
                checkpoint = torch.load(pt_path, map_location=torch.device('cpu'), weights_only=False)
        
        # Extract the actual model if using direct load
        if 'model' not in locals():  # If model wasn't set by attempt_load
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    model = checkpoint['model']
                elif 'ema' in checkpoint:
                    model = checkpoint['ema']
                else:
                    print("Error: Checkpoint doesn't contain model or EMA")
                    return False
            else:
                model = checkpoint
        
        # Convert model to float32 precision to avoid "expected scalar type Float but found Half" error
        model = model.to(torch.float32)
        
        # Set to evaluation mode
        model.eval()
        
        # Create a wrapper class to handle the YOLOv7 model outputs
        class YOLOv7Wrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                # Set model to inference mode
                self.model.eval()

            def forward(self, x):
                # YOLOv7 inference returns different formats
                # This wrapper ensures consistent output
                out = self.model(x)
                if isinstance(out, tuple):
                    return out[0]  # Return only first element (detections)
                return out

        # Wrap the model for consistent output
        wrapper_model = YOLOv7Wrapper(model)
        
        # Run once to initialize all parameters
        dummy_input = torch.zeros((1, 3, 640, 640), device='cpu')
        with torch.no_grad():
            _ = wrapper_model(dummy_input)
        
        # Trace the model
        traced_model = torch.jit.trace(wrapper_model, dummy_input)
        
        # Save the TorchScript model
        traced_model.save(output_path)
        print(f"TorchScript model saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_model.py <input_pt_file> <output_torchscript_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist")
        sys.exit(1)
    
    if convert_pt_to_torchscript(input_file, output_file):
        print("Conversion successful!")
    else:
        print("Conversion failed.")
        sys.exit(1) 