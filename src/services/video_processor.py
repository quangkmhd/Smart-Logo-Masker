import os
import cv2
import numpy as np
import subprocess
import logging
from ultralytics import YOLO
from src.core.config import settings
from src.schemas.task import ProcessingOptions

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self):
        self.model = self._load_model()

    def _load_model(self):
        model_path = settings.MODEL_PATH
        if not os.path.exists(model_path):
            logger.warning(f"Model {model_path} not found. Using fallback {settings.FALLBACK_MODEL}")
            model_path = settings.FALLBACK_MODEL
        
        try:
            return YOLO(model_path)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None

    def process(self, input_path: str, output_path: str, options: ProcessingOptions = None) -> bool:
        if self.model is None:
            logger.error("Model not loaded, cannot process video.")
            return False
            
        opts = options or ProcessingOptions()

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {input_path}")
            return False

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps:
            fps = 30

        # Temporary output before ffmpeg conversion
        temp_output = output_path + ".tmp.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Starting processing: {total_frames} frames with options {opts}")

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Inference
            results = self.model.predict(
                frame, 
                conf=opts.conf, 
                iou=opts.iou, 
                classes=opts.target_classes,
                verbose=False
            )
            result = results[0]

            # Apply masking
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                for mask in masks:
                    mask_resized = cv2.resize(mask, (width, height))
                    mask_resized = np.expand_dims(mask_resized, axis=-1)
                    
                    if opts.mask_mode == "blur":
                        # Ensure kernel size is odd
                        k = opts.blur_intensity if opts.blur_intensity % 2 != 0 else opts.blur_intensity + 1
                        blurred_frame = cv2.GaussianBlur(frame, (k, k), 0)
                        frame = np.where(mask_resized > 0.5, blurred_frame, frame).astype(np.uint8)
                    else: # solid
                        frame = np.where(mask_resized > 0.5, 0, frame).astype(np.uint8)
                        
            elif result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        if opts.mask_mode == "blur":
                            k = opts.blur_intensity if opts.blur_intensity % 2 != 0 else opts.blur_intensity + 1
                            roi = cv2.GaussianBlur(roi, (k, k), 0)
                            frame[y1:y2, x1:x2] = roi
                        else: # solid
                            frame[y1:y2, x1:x2] = 0

            out.write(frame)
            frame_count += 1
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count}/{total_frames} frames")

        cap.release()
        out.release()

        # Convert to H.264 for web compatibility
        logger.info("Converting to H.264...")
        try:
            subprocess.run([
                "ffmpeg", "-y", 
                "-i", temp_output, 
                "-i", input_path, 
                "-map", "0:v", 
                "-map", "1:a?", 
                "-c:v", "libx264", 
                "-c:a", "aac", 
                "-shortest",
                output_path
            ], check=True, capture_output=True)
            if os.path.exists(temp_output):
                os.remove(temp_output)
            return True
        except Exception as e:
            logger.error(f"FFmpeg conversion failed: {e}")
            # Fallback: rename temp to output if ffmpeg fails
            if os.path.exists(temp_output):
                os.rename(temp_output, output_path)
            return True

video_processor = VideoProcessor()
