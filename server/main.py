import base64
import io
import logging
import time
import asyncio # Required for WebSocket handling
from typing import List, Dict, Any

import cv2
import numpy as np
from deepface import DeepFace
from fastapi import FastAPI, HTTPException, Body, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Facial Emotion Detection API with WebSocket",
    description="API for real-time facial emotion detection using DeepFace, with WebSocket support for frame streaming.",
    version="1.1.0",
)

# Configure CORS
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models (can be reused or adapted if needed for WebSocket messages) ---
class EmotionPrediction(BaseModel):
    region: Dict[str, int]
    dominant_emotion: str
    emotion_probabilities: Dict[str, float]

class EmotionAnalysisResponse(BaseModel):
    predictions: List[EmotionPrediction]
    processing_time_ms: float
    message: str = "Emotion analysis successful"
    error: bool = False
    error_message: str = None


# --- Helper Functions ---
def base64_to_cv2_image(base64_string: str):
    try:
        if "," in base64_string:
            header, encoded_data = base64_string.split(",", 1)
        else:
            encoded_data = base64_string
        
        decoded_bytes = base64.b64decode(encoded_data)
        image_array = np.frombuffer(decoded_bytes, dtype=np.uint8)
        cv2_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if cv2_image is None:
            raise ValueError("Failed to decode image.")
        return cv2_image
    except Exception as e:
        logger.error(f"Error decoding base64 image: {e}")
        return None # Return None to handle error upstream

# --- WebSocket Endpoint for Emotion Detection ---
@app.websocket("/ws/emotion_detection")
async def websocket_emotion_detection(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established.")
    try:
        while True:
            # Receive frame data URL from the client
            data_url = await websocket.receive_text()
            start_time = time.time()

            if not data_url:
                continue

            cv2_image = base64_to_cv2_image(data_url)

            if cv2_image is None:
                response = EmotionAnalysisResponse(
                    predictions=[],
                    processing_time_ms=(time.time() - start_time) * 1000,
                    message="Failed to decode image from data URL.",
                    error=True,
                    error_message="Image decoding failed on server."
                )
                await websocket.send_json(response.dict())
                continue
            
            predictions_list: List[EmotionPrediction] = []
            try:
                # DeepFace analysis
                analysis_results = DeepFace.analyze(
                    img_path=cv2_image,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='opencv' # Using 'opencv' for potentially faster processing in real-time stream
                                              # Other options: 'mtcnn', 'ssd', 'retinaface' (might be slower)
                )

                if analysis_results and isinstance(analysis_results, list):
                    for face_data in analysis_results:
                        if isinstance(face_data, dict):
                            region = face_data.get("region")
                            dominant_emotion = face_data.get("dominant_emotion")
                            emotion_probs = face_data.get("emotion")

                            if region and dominant_emotion and emotion_probs:
                                formatted_region = {
                                    "x": int(region.get("x", 0)),
                                    "y": int(region.get("y", 0)),
                                    "w": int(region.get("w", 0)),
                                    "h": int(region.get("h", 0)),
                                }
                                formatted_emotion_probs = {
                                    k: float(v) for k, v in emotion_probs.items()
                                }
                                predictions_list.append(
                                    EmotionPrediction(
                                        region=formatted_region,
                                        dominant_emotion=str(dominant_emotion),
                                        emotion_probabilities=formatted_emotion_probs,
                                    )
                                )
                
                processing_time = (time.time() - start_time) * 1000
                if not predictions_list and analysis_results: # analysis happened but no faces
                     response = EmotionAnalysisResponse(
                        predictions=[],
                        processing_time_ms=processing_time,
                        message="No faces detected in the frame."
                    )
                elif not predictions_list and not analysis_results: # analysis failed or returned empty
                    response = EmotionAnalysisResponse(
                        predictions=[],
                        processing_time_ms=processing_time,
                        message="Analysis returned no results."
                    )
                else:
                    response = EmotionAnalysisResponse(
                        predictions=predictions_list,
                        processing_time_ms=processing_time,
                        message=f"Analysis successful, {len(predictions_list)} face(s) found."
                    )

            except Exception as e:
                logger.error(f"DeepFace analysis failed during WebSocket processing: {e}")
                processing_time = (time.time() - start_time) * 1000
                response = EmotionAnalysisResponse(
                    predictions=[],
                    processing_time_ms=processing_time,
                    message="Error during emotion analysis.",
                    error=True,
                    error_message=str(e)
                )
            
            await websocket.send_json(response.dict())

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket connection: {e}")
        try:
            # Try to inform client about the error before closing if possible
            await websocket.send_json(EmotionAnalysisResponse(
                predictions=[], processing_time_ms=0, message="An unexpected server error occurred.", error=True, error_message=str(e)
            ).dict())
        except: # If sending fails, it means connection is already too broken
            pass
        await websocket.close()


# --- Root endpoint for basic info ---
@app.get("/", tags=["General"])
async def read_root():
    return {
        "message": "Welcome to the Facial Emotion Detection API!",
        "version": app.version,
        "documentation": "/docs",
        "websocket_endpoint": "/ws/emotion_detection"
    }

# Note: The previous HTTP POST endpoint (/analyze_emotion) can be kept for single image analysis
# or removed if WebSocket is the sole method for real-time. For this example, I'll keep it.
# (Code for ImageInput and analyze_emotion_endpoint from previous version would go here if kept)
# --- Pydantic Models for Request and Response (for HTTP POST endpoint) ---
class ImageInput(BaseModel):
    image_data_url: str

class HTTPEmotionResponse(BaseModel): # Renamed to avoid conflict if structure differs
    predictions: List[EmotionPrediction]
    processing_time_ms: float
    message: str = "Emotion analysis successful"

@app.post("/analyze_emotion_http", response_model=HTTPEmotionResponse, tags=["Emotion Detection (HTTP)"])
async def analyze_emotion_http_endpoint(image_input: ImageInput = Body(...)):
    start_time = time.time()
    logger.info("Received request for HTTP emotion analysis.")
    # ... (rest of the HTTP POST endpoint implementation from the previous version) ...
    # This is just a placeholder to show it can co-exist.
    # For brevity, I'm not repeating the full code from the previous version here.
    # You would copy the logic from the previous `analyze_emotion_endpoint`.
    # Ensure it uses `HTTPEmotionResponse` and `EmotionPrediction` models.
    # Example minimal response:
    cv2_image = base64_to_cv2_image(image_input.image_data_url)
    if cv2_image is None:
        raise HTTPException(status_code=400, detail="Could not decode image from base64 string.")
    
    # Dummy analysis for placeholder
    processing_time = (time.time() - start_time) * 1000
    return HTTPEmotionResponse(
        predictions=[], 
        processing_time_ms=processing_time,
        message="HTTP endpoint needs full implementation if used."
    )


if __name__ == "__main__":
    import uvicorn
    # This part is for running directly, e.g. `python main.py`
    # Usually, you'd run with `uvicorn main:app --reload --host 0.0.0.0 --port 8000`
    uvicorn.run(app, host="0.0.0.0", port=8000)