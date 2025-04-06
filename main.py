import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict
import logging

# Import the detector class (assuming it's in the model directory)
try:
    from model.model_runner import DeepFakeDetector
except ImportError:
    logging.error("Could not import DeepFakeDetector. Make sure model/model_runner.py exists.")
    # Define a placeholder if import fails to allow app startup, but endpoints will fail
    class DeepFakeDetector:
        def __init__(self, *args, **kwargs):
            logging.warning("Using placeholder DeepFakeDetector due to import error.")
        def predict(self, base64_image: str):
            return {"error": "Detector not loaded due to import error"}

# --- Pydantic Models ---

class PredictRequest(BaseModel):
    """Request model for the predict endpoint"""
    image: str = Field(..., description="Base64 encoded image string")

class PredictResponse(BaseModel):
    """Response model for the predict endpoint"""
    prediction: str = Field(..., description="Predicted class label (e.g., 'REAL', 'FAKE')")
    confidence: float = Field(..., description="Confidence score of the prediction (0.0 to 1.0)")
    probabilities: Dict[str, float] = Field(..., description="Probability distribution for each class")

class HealthResponse(BaseModel):
    """Response model for the health check endpoint"""
    status: str = "ok"
    service: str = "deepfake-detector-api"

# --- FastAPI App Setup ---

app = FastAPI(
    title="DeepFake Detector API",
    description="API for detecting deepfakes in images using a PyTorch model.",
    version="1.0.0",
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Global Detector Instance ---
# Instantiate the detector globally. FastAPI handles the lifecycle.
# Consider dependency injection for more complex scenarios or testing.
try:
    detector = DeepFakeDetector() # Assumes model files are in ./model/model_export/
    logging.info("DeepFakeDetector loaded successfully.")
except Exception as e:
    original_error_message = str(e)
    logging.error(f"Failed to instantiate DeepFakeDetector: {original_error_message}", exc_info=True)
    # Create a placeholder that returns errors if instantiation fails
    class ErrorDetector:
        def __init__(self, error_message: str):
            # Store the error message from instantiation time
            self.error_message = f"Detector instantiation failed: {error_message}"

        def predict(self, base64_image: str):
            # Return the stored error message
            return {"error": self.error_message}
    # Pass the original error message to the ErrorDetector instance
    detector = ErrorDetector(original_error_message)


# --- API Endpoints ---

@app.get("/", tags=["General"])
async def read_root():
    """
    Root endpoint providing a welcome message.
    """
    return {"message": "Welcome to the DeepFake Detector API!"}

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """
    Health check endpoint to verify service availability.
    """
    return HealthResponse()


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_deepfake(request: PredictRequest):
    """
    Accepts a base64 encoded image and returns deepfake prediction results.
    """
    try:
        # Call the detector's predict method
        result = detector.predict(request.image)

        # Check if the detector returned an error
        if "error" in result:
            logging.error(f"Prediction error from detector: {result['error']}")
            raise HTTPException(
                status_code=500,
                detail=f"Model prediction failed: {result['error']}",
            )

        # Validate and structure the response using Pydantic models
        # Pydantic will implicitly validate the structure based on PredictResponse
        return PredictResponse(**result)

    except HTTPException as http_exc:
        # Re-raise HTTPException to let FastAPI handle it
        raise http_exc
    except Exception as e:
        logging.error(f"Unexpected error during prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}",
        )

# --- Uvicorn Runner ---
# (Optional: For running directly with `python main.py`)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080) 