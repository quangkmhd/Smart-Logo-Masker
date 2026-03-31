# Hard Negative Miner 🔍

This document guides you on how to use the **Hard Negative Miner** tool to collect "hard" background data (cases where the model predicts incorrectly) to improve accuracy and reduce the False Positive rate for your YOLO model.

## 1. Introduction

During the logo detection model training process, the model often encounters cases where it misidentifies other objects as logos (False Positives).

This tool automates the following workflow:
1.  **Download videos** from target YouTube channels (e.g., CNN, sports channels).
2.  **Run predictions** using the current model on the downloaded videos.
3.  **Collect frames** where the model predicts a logo is present but might actually be wrong (Hard Negatives). Collecting these frames provides a high-quality background dataset to retrain the model, making it more robust.

## 2. Output Directory Structure

After running, the tool creates the following directories:
- `downloads/`: Contains temporary video files downloaded from YouTube (automatically deleted after processing to save storage).
- `out/`: Contains processing results:
    - `{video_name}/raw/`: Contains the extracted original frames.
    - `{video_name}/pred/`: Contains frames with drawn prediction masks (for visual inspection).
- `download_predict_log.csv`: A log file to track the processing progress.

## 3. How to Use

### Configuration
Open `src/services/hard_negative_miner.py` and adjust the parameters in the `main()` function:

```python
target_frames = 1000 # Target number of frames to collect
input_url = "https://www.youtube.com/@CNN/videos" # Video source
model_path = "path/to/your/model.pt" # Path to your current model
```

### Running the tool
```bash
python3 src/services/hard_negative_miner.py
```

## 4. Performance Optimization

The tool is designed with a multi-threading mechanism to simultaneously optimize Downloading and Prediction:
- **Thread 1 (Downloader):** Downloads videos into a queue. Uses `yt-dlp` with options optimized for speed and to avoid being blocked by YouTube.
- **Thread 2 (Predictor):** Retrieves videos from the queue and runs predictions using the GPU.

## 5. Model Improvement Workflow (Retraining)

Once you have the images in the `out/` directory, you can:
1.  Inspect images in the `pred` folder to identify cases where the model predicted incorrectly.
2.  Take the corresponding original images from the `raw` folder.
3.  Add these images to the training dataset as **Background images** (with no labels or an empty label file).
4.  Proceed with retraining (Retrain/Fine-tune) so the model learns to distinguish between real logos and these confusing objects.
