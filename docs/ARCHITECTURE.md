# System Architecture of Smart Logo Masker

## 1. Introduction

Smart Logo Masker is an end-to-end computer vision pipeline built to automatically detect, segment, and obscure (blur or black-out) specific target logos in video streams. The system is designed for high-throughput video processing, catering specifically to broadcasters and content moderators who need to remove unauthorized betting or sponsor logos from live or pre-recorded media.

The architecture is split into two primary workflows:
1. **The Data Preparation & Training Pipeline**: For fine-tuning the model on new, custom logos.
2. **The Inference & Masking Pipeline**: For real-time or batch processing of video files.

## 2. Core Components

### 2.1. The Detection & Segmentation Engine (YOLOv26-seg)
The core of the system relies on an advanced instance segmentation model, specifically a custom implementation/variant referred to in this project as `YOLOv26-seg`.
- **Why Instance Segmentation?** Standard object detection (bounding boxes) is insufficient because betting logos are often irregular in shape or placed diagonally on jerseys. Bounding boxes would blur too much of the background action (e.g., a player's face or the ball). Instance segmentation provides a pixel-perfect polygon mask around the logo contour.
- **Model Output**: For every frame, the model outputs bounding box coordinates, class probabilities, and a set of mask coefficients that are combined with prototype masks to generate the final binary mask for the target logo.

### 2.2. The Data Ingestion Module (`prepare_data.py`)
To train the model on new logos, users provide images annotated with polygons using the open-source `LabelMe` tool.
- **JSON Parsing**: The script reads the LabelMe `.json` files, extracting image dimensions and polygon point arrays.
- **Coordinate Normalization**: YOLO requires polygon coordinates to be normalized between `0.0` and `1.0`. The script mathmatically maps absolute pixel coordinates to relative ones.
- **Dataset Splitting**: It automatically shuffles and partitions the raw data into `train` and `val` (validation) sets, creating the rigid directory structure expected by the YOLO training engine.
- **YAML Generation**: It dynamically generates the `data.yaml` file containing the paths and class names.

### 2.3. The Inference Pipeline (`main.py` / `app.py`)
This is the execution engine that processes video files.
- **Frame Extraction**: Utilizes OpenCV (`cv2.VideoCapture`) to read frames sequentially.
- **Batching & Inference**: Frames are resized to the configured inference size (e.g., 640x640) and pushed through the YOLO network.
- **Mask Application**:
  - The model returns a binary mask where pixels belonging to the logo are `True`.
  - The pipeline extracts this region of interest (ROI) from the original high-resolution frame.
  - A Gaussian blur (`cv2.GaussianBlur`) or pixelation algorithm is applied specifically to the ROI.
  - The blurred ROI is composited back onto the original frame using the mask as an alpha channel, ensuring seamless integration without bleeding into the background.
- **Video Writing**: The modified frames are encoded back into an `.mp4` file using `cv2.VideoWriter`.

## 3. Deployment Architecture

To ensure environmental consistency, the system is fully containerized.
- **Docker Compose**: Orchestrates the environment. It builds an image based on an NVIDIA CUDA base image, installs system-level dependencies (`libgl1-mesa-glx` for OpenCV), and installs Python requirements.
- **Volume Mounting**: The `/asset` and `/runs` directories are mapped to the host machine so that input videos can be dropped in and output videos/model weights can be retrieved without entering the container.

## 4. Design Trade-offs

- **Speed vs. Accuracy**: The architecture uses a single-stage YOLO model rather than a two-stage Mask R-CNN. While Mask R-CNN might yield marginally better mask edges on highly complex shapes, YOLO operates at least 3x faster, which is an absolute requirement for processing 30fps/60fps HD video feeds.
- **Hardcoded Augmentations**: The training pipeline relies heavily on mosaic and mixup augmentations (inherent to YOLO) because betting logos often appear small and distorted on player shirts. This makes the model robust but increases the required training epochs.
