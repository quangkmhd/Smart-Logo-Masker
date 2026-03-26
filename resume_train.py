from ultralytics import YOLO

def resume_training():
    # Path to the last checkpoint saved
    last_weights_path = "runs/segment/runs/segment/logo_masker_model/weights/last.pt"
    
    print(f"[*] Resuming training from: {last_weights_path}")
    
    # Load the checkpoint
    model = YOLO(last_weights_path)
    
    # Resume training
    # YOLO automatically reads the training parameters from the checkpoint's attributes
    results = model.train(resume=True)
    
    print("Training is complete! Check the runs directory for weights and results.")

if __name__ == "__main__":
    resume_training()
