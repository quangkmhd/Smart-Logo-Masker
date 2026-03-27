import os
from ultralytics import YOLO

def train():
    # Load YOLO11 segmentation model (pretrained)
    # Using 'yolo11n-seg.pt' as the base, you can change to 's', 'm', 'l', 'x' if desired
    model = YOLO("yolo26x-seg.pt")

    # Path to your dataset YAML
    data_yaml_path = "/home/quangnhvn34/dev/me/Smart-Logo-Masker/dataset/data.yaml"

    # The models are trained with product-standard hyperparameter setup
    results = model.train(
        data=data_yaml_path,
        name="logo_masker_model",  # Experiment name
        exist_ok=True,             # Overwrite existing experiment folder if exists
        epochs=300,               # Max training epochs
        patience=50,               # Early stopping (stop if no improvement after 50 epochs)
        imgsz=640,                 # Image resolution
        batch=4,                   # Batch size (8GB VRAM friendly)
        device=0,                  # GPU id
        workers=8,                 # Dataloader workers (speed up data loading)
        
        # Optimizer and scheduler settings
        optimizer="auto",          # Automatically chooses best optimizer (AdamW usually)
        lr0=0.01,                  # Initial learning rate
        lrf=0.01,                  # Final OneCycleLR learning rate (lr0 * lrf)
        weight_decay=0.0005,       # Optimizer weight decay
        momentum=0.937,            # Optimizer momentum
        
        # Training behavior
        save=True,                 # Save model checkpoints
        save_period=-1,            # Model saving frequency (-1 to save only best and last)
        val=True,                  # Run validation at the end of every epoch
        
        # Augmentations (to make the model robust)
        hsv_h=0.015,               # Image HSV-Hue augmentation (fraction)
        hsv_s=0.7,                 # Image HSV-Saturation augmentation (fraction)
        hsv_v=0.4,                 # Image HSV-Value augmentation (fraction)
        degrees=0.0,               # Image rotation (+/- deg)
        translate=0.1,             # Image translation (+/- fraction)
        scale=0.5,                 # Image scale (+/- gain)
        shear=0.0,                 # Image shear (+/- deg)
        perspective=0.0,           # Image perspective (+/- fraction)
        flipud=0.0,                # Image flip up-down probability
        fliplr=0.5,                # Image flip left-right probability
        bgr=0.0,                   # Image channel BGR probability
        mosaic=1.0,                # Mosaic augmentation probability
        mixup=0.1,                 # Mixup augmentation probability
        copy_paste=0.1             # Copy-paste augmentation probability (excellent for segmentation)
    )

    print("Training is complete! Check the 'runs/segment/logo_masker_model' directory for weights and results.")

if __name__ == "__main__":
    train()
