# AI WROTE THIS SHIT, I'VE NO IDEA HOW IT WORKS, AND WHY IT WORKS
# I'M NOT EVEN SURE IF THIS IS CORRECT
import torch
import torch.nn as nn
import timm # Import timm
import json
import base64
from PIL import Image
import io
import torchvision.transforms as transforms
import os
import logging # Add logging

class DeepFakeDetector:
    def __init__(self, model_dir="model/model_export"):
        try:
            # Load model info
            with open(os.path.join(model_dir, "model_info.json"), "r") as f:
                self.model_info = json.load(f)

            num_classes = len(self.model_info["class_mapping"])
            model_path = os.path.join(model_dir, "deepfake_detector.pth")
            device = torch.device('cpu') # Explicitly use CPU
            logging.info(f"Loading model info: {self.model_info}")
            logging.info(f"Number of classes: {num_classes}")
            logging.info(f"Model path: {model_path}")

            # 1. Instantiate the ConvNeXt Tiny architecture using TIMM
            logging.info(f"Instantiating TIMM ConvNeXt Tiny architecture (convnext_tiny) with {num_classes} classes...")
            # Use timm.create_model, pretrained=False as we load weights, specify num_classes
            model_arch = timm.create_model('convnext_tiny', pretrained=False, num_classes=num_classes)
            logging.info("TIMM Architecture instantiated.")

            # 2. Load the full checkpoint dictionary
            logging.info(f"Loading checkpoint dictionary from {model_path}...")
            checkpoint = torch.load(model_path, map_location=device)
            logging.info("Checkpoint dictionary loaded.")

            # 4. Extract the state dict (assuming it's under the 'model' key, as indicated by error logs)
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
                logging.info("Extracted state_dict from checkpoint['model'].")
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                 state_dict = checkpoint['state_dict'] # Common alternative key
                 logging.info("Extracted state_dict from checkpoint['state_dict'].")
            elif isinstance(checkpoint, dict):
                 # Check if checkpoint itself is the state_dict
                 first_key = list(checkpoint.keys())[0]
                 if first_key.startswith(('stem.', 'stages.', 'head.', 'norm.', 'classifier.')): # Direct timm match
                      state_dict = checkpoint
                      logging.info("Checkpoint dictionary keys match timm structure, assuming it's the state_dict directly.")
                 elif first_key.startswith('model.'): # Match with 'model.' prefix
                      state_dict = checkpoint
                      logging.info("Checkpoint dictionary keys start with 'model.', assuming it's the state_dict but needs cleaning.")
                 else:
                      logging.error(f"Checkpoint dictionary does not contain expected keys ('model' or 'state_dict') and keys don't match expected prefixes ('stem.', 'model.'). Keys: {list(checkpoint.keys())[:5]}")
                      raise KeyError("Could not find the model state_dict within the loaded checkpoint or recognize key format.")
            elif isinstance(checkpoint, OrderedDict):
                # Check keys again for prefix
                first_key = list(checkpoint.keys())[0]
                if first_key.startswith('model.'):
                    state_dict = checkpoint
                    logging.info("Loaded OrderedDict keys start with 'model.', assuming it's the state_dict but needs cleaning.")
                elif first_key.startswith(('stem.', 'stages.', 'head.', 'norm.', 'classifier.')):
                    state_dict = checkpoint
                    logging.info("Loaded OrderedDict keys match timm structure, assuming it's the state_dict directly.")
                else:
                    # Treat as state_dict but warn about unexpected keys
                    state_dict = checkpoint
                    logging.warning(f"Loaded OrderedDict keys do not start with expected prefixes. Proceeding, but check key matching. Keys: {list(checkpoint.keys())[:5]}")
            else:
                logging.error(f"Loaded checkpoint is not in a recognized format (dict or OrderedDict). Type: {type(checkpoint)}")
                raise TypeError("Loaded checkpoint file is not in the expected format.")

            # 5. Clean the state dict keys (handle potential 'module.' or 'model.' prefix)
            if list(state_dict.keys()): # Check if state_dict is not empty
                first_key = list(state_dict.keys())[0]
                if first_key.startswith('module.'):
                    logging.info("Removing 'module.' prefix from state dict keys...")
                    state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
                elif first_key.startswith('model.'):
                    logging.info("Removing 'model.' prefix from state dict keys...")
                    state_dict = {k[len('model.'):]: v for k, v in state_dict.items()}
            else:
                logging.warning("State dict appears to be empty after extraction.")

            # 5b. Rename the final classifier layer keys to match timm expectation
            if 'fier.weight' in state_dict and 'fier.bias' in state_dict:
                logging.info("Renaming final layer keys: 'fier.weight' -> 'head.fc.weight', 'fier.bias' -> 'head.fc.bias'")
                state_dict['head.fc.weight'] = state_dict.pop('fier.weight')
                state_dict['head.fc.bias'] = state_dict.pop('fier.bias')
            elif 'classifier.weight' in state_dict and 'classifier.bias' in state_dict:
                # Handle case where it might have been saved with 'classifier' name directly
                logging.info("Renaming final layer keys: 'classifier.weight' -> 'head.fc.weight', 'classifier.bias' -> 'head.fc.bias'")
                state_dict['head.fc.weight'] = state_dict.pop('classifier.weight')
                state_dict['head.fc.bias'] = state_dict.pop('classifier.bias')
            else:
                # If neither 'fier' nor 'classifier' keys are found, check if 'head.fc' already exists (maybe loaded correctly)
                 if not ('head.fc.weight' in state_dict and 'head.fc.bias' in state_dict):
                     logging.warning("Could not find expected final layer keys ('fier.*' or 'classifier.*') to rename to 'head.fc.*'. Loading might fail if head.fc keys are also missing.")

            # 6. Load the cleaned and potentially renamed state dict into the TIMM architecture
            logging.info("Comparing keys before loading...")
            model_keys = set(model_arch.state_dict().keys())
            dict_keys = set(state_dict.keys())

            # Print a few keys for comparison
            logging.debug(f"Sample model keys: {list(model_keys)[:5]}")
            logging.debug(f"Sample state_dict keys: {list(dict_keys)[:5]}")

            # Print differences (optional, can be long)
            # logging.debug(f"Keys in model but not state_dict: {model_keys - dict_keys}")
            # logging.debug(f"Keys in state_dict but not model: {dict_keys - model_keys}")

            logging.info("Attempting to load state dict into TIMM model architecture (strict=True)...")
            try:
                load_result = model_arch.load_state_dict(state_dict, strict=True)
                logging.info("State dict loaded successfully into TIMM architecture.")
            except RuntimeError as load_error:
                logging.error(f"RUNTIME ERROR during strict state_dict loading: {load_error}")
                # Re-raising the error to halt initialization, as it's critical
                raise load_error

            # 7. Assign the loaded model and set to evaluation mode
            self.model = model_arch.to(device)
            self.model.eval()
            logging.info("Model assigned and set to evaluation mode.")

            # Set up image transform pipeline
            self.transform = transforms.Compose([
                transforms.Resize((self.model_info["input_size"][0], self.model_info["input_size"][1])),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.model_info["normalize_mean"],
                    std=self.model_info["normalize_std"]
                )
            ])
            logging.info("Image transform pipeline configured.")

            # Class mapping (reverse for prediction output)
            self.class_mapping = {v: k for k, v in self.model_info["class_mapping"].items()}
            logging.info(f"Class mapping configured: {self.class_mapping}")

        except Exception as init_e:
            logging.error(f"Error during DeepFakeDetector initialization: {init_e}", exc_info=True)
            # Re-raise the exception to prevent the ErrorDetector from masking it
            # in the FastAPI app startup logs.
            raise init_e


    def predict(self, base64_image):
        try:
            # Decode base64 image - handle data URL prefix
            if ";base64," in base64_image:
                header, encoded = base64_image.split(";base64,", 1)
            else:
                # Assume it's just the base64 string if no prefix
                encoded = base64_image

            image_data = base64.b64decode(encoded)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')

            # Preprocess the image
            image_tensor = self.transform(image).unsqueeze(0)

            # Run inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]

                # Get prediction index and confidence
                confidence, pred_idx = torch.max(probs, 0)
                pred_idx_item = pred_idx.item()
                confidence_item = confidence.item()

                # Format probabilities dictionary
                probs_list = probs.tolist()
                probabilities_dict = {
                    self.class_mapping.get(i, f"unknown_class_{i}"): float(p)
                    for i, p in enumerate(probs_list)
                }

                # Format results
                result = {
                    "prediction": self.class_mapping.get(pred_idx_item, f"unknown_class_{pred_idx_item}"),
                    "confidence": float(confidence_item),
                    "probabilities": probabilities_dict
                }
                # logging.info(f"Prediction result: {result}") # Optional: Log successful prediction
                return result
        except Exception as e:
            # Log the detailed error for debugging
            import traceback
            logging.error(f"Error during prediction: {e}\n{traceback.format_exc()}")
            # Return error structure consistent with successful prediction but with an error key
            return {"error": f"Prediction failed: {str(e)}"}

# For testing outside of Rust
if __name__ == "__main__":
    import sys
    from collections import OrderedDict # Add this import if needed
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        logging.info(f"Running test prediction for image: {image_path}")
        try:
            with open(image_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
                # Optional: Add data URL prefix for consistency during testing
                # img_data = f"data:image/jpeg;base64,{img_data}"

                # Instantiate detector
                detector = DeepFakeDetector()
                logging.info("Detector initialized for testing.")

                # Perform prediction
                result = detector.predict(img_data)

                # Print result
                print(json.dumps(result, indent=2))

        except FileNotFoundError:
            logging.error(f"Error: Test image file not found at {image_path}")
        except Exception as main_e:
            logging.error(f"Error in __main__ test execution: {main_e}", exc_info=True)
    else:
        print("Usage: python model_runner.py <path_to_image_file>")