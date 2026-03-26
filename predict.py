import os
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Smart Logo Masker - Production Inference Script")
    parser.add_argument("--source", type=str, required=True, help="Path to input image, video, directory, or stream URL")
    parser.add_argument("--weights", type=str, default="runs/segment/logo_masker_model/weights/best.pt", help="Path to the trained YOLO model weights")
    parser.add_argument("--conf", type=float, default=0.75, help="Confidence threshold for predictions")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image sizes")
    parser.add_argument("--device", type=str, default="0", help="cuda device, i.e. 0, 1, 2, 3 or cpu")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum number of detections per image")
    parser.add_argument("--vid-stride", type=int, default=1, help="Video frame-rate stride")
    parser.add_argument("--retina-masks", action="store_true", help="Use high-resolution segmentation masks (slower but more accurate edges)")
    parser.add_argument("--agnostic-nms", action="store_true", help="Enable class-agnostic NMS")
    parser.add_argument("--save-txt", action="store_true", help="Save results to *.txt")
    parser.add_argument("--save-crop", action="store_true", help="Save cropped prediction boxes")
    parser.add_argument("--project", type=str, default="runs/predict", help="Save results to project/name")
    parser.add_argument("--name", type=str, default="logo_inference", help="Save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="Existing project/name ok, do not increment")
    parser.add_argument("--only-detected", action="store_true", help="Only save images where at least one detection occurred")

    
    return parser.parse_args()

def infer(args):
    # Verify input source
    if not os.path.exists(args.source) and not args.source.startswith(('http://', 'https://', 'rtsp://', 'rtmp://')):
        print(f"Error: Source '{args.source}' does not exist.")
        sys.exit(1)

    # Load the model properly with error handling
    try:
        print(f"[*] Loading model from {args.weights} ...")
        model = YOLO(args.weights)
    except Exception as e:
        print(f"[!] Failed to load model '{args.weights}': {e}")
        print("[!] Attempting to fallback to a pre-trained base model 'yolo11n-seg.pt'...")
        try:
            model = YOLO("yolo11n-seg.pt")
        except Exception as fallback_err:
            print(f"Error: Fallback model could not be loaded: {fallback_err}")
            sys.exit(1)

    # Run Prediction with advanced parameters
    print(f"[*] Starting inference on: {args.source}")
    
    # Run the prediction
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device if args.device else None,
        max_det=args.max_det,
        vid_stride=args.vid_stride,
        retina_masks=args.retina_masks,
        agnostic_nms=args.agnostic_nms,
        save=not args.only_detected,  # If only_detected is True, we save manually later
        save_txt=args.save_txt,
        save_crop=args.save_crop,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        line_width=2,            # UI setting for bounding box thickness
        show_labels=True,        # Show class labels
        show_conf=True,          # Show confidence scores
        stream=True              # Use stream mode for large directories to save memory
    )
    
    # Process Results
    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    detected_count = 0
    total_count = 0
    
    print(f"[*] Processing results and saving to {save_dir}...")
    
    for result in results:
        total_count += 1
        has_detection = len(result.boxes) > 0
        
        if args.only_detected:
            if has_detection:
                detected_count += 1
                # Save manually to the target directory
                # result.save() will save with standard YOLO visualization
                # For images, we can use the original filename
                p = Path(result.path)
                save_path = save_dir / p.name
                result.save(filename=str(save_path))
        else:
            if has_detection:
                detected_count += 1
            # If save=True was set in predict(), the images are already saved by YOLO.
            # But if stream=True, we might need to handle it or if we want custom naming.
            pass

    print(f"\n[*] Inference complete!")
    print(f"[*] Total processed: {total_count}")
    print(f"[*] Detections found in: {detected_count} files")
    print(f"[*] Results saved to: {save_dir}")


if __name__ == "__main__":
    args = parse_args()
    infer(args)
