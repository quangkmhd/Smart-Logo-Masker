import os
import cv2
import tempfile
import numpy as np
import subprocess
import gradio as gr
from ultralytics import YOLO

# Đoạn mã Javascript để điều khiển phát video cùng lúc
js_play = """
function() {
    var v1 = document.querySelector('#vid1 video');
    var v2 = document.querySelector('#vid2 video');
    if (v1 && v2) {
        v1.play();
        v2.play();
    }
    return [];
}
"""

# Đoạn mã Javascript để điều khiển dừng video cùng lúc
js_pause = """
function() {
    var v1 = document.querySelector('#vid1 video');
    var v2 = document.querySelector('#vid2 video');
    if (v1 && v2) {
        v1.pause();
        v2.pause();
    }
    return [];
}
"""

# Khởi tạo model YOLO
model_path = "runs/segment/logo_masker_model/weights/best.pt"
if not os.path.exists(model_path):
    print(f"Warning: Model {model_path} không tồn tại. Thử dùng yolo11n-seg.pt")
    model_path = "yolo11n-seg.pt"

try:
    model = YOLO(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def process_video(input_video_path, conf, iou, blur_intensity, mask_mode):
    if not input_video_path or model is None:
        return input_video_path
        
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps:
        fps = 30
        
    temp_dir = tempfile.mkdtemp()
    temp_output = os.path.join(temp_dir, "temp_out.mp4")
    final_output = os.path.join(temp_dir, "masked_output.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Bắt đầu xử lý: {total_frames} frames, Conf: {conf}, IoU: {iou}, Blur: {blur_intensity}, Mode: {mask_mode}")
    
    # Đảm bảo blur_intensity là số lẻ
    if blur_intensity % 2 == 0:
        blur_intensity += 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model.predict(frame, conf=conf, iou=iou, verbose=False)
        result = results[0]
        
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            for mask in masks:
                mask_resized = cv2.resize(mask, (width, height))
                
                if mask_mode == "blur":
                    blurred_frame = cv2.GaussianBlur(frame, (blur_intensity, blur_intensity), 0)
                    mask_resized = np.expand_dims(mask_resized, axis=-1)
                    frame = np.where(mask_resized > 0.5, blurred_frame, frame).astype(np.uint8)
                else: # solid
                    solid_color = np.full_like(frame, (30, 30, 30)) 
                    mask_resized = np.expand_dims(mask_resized, axis=-1)
                    frame = np.where(mask_resized > 0.5, solid_color, frame).astype(np.uint8)
                    
        elif result.boxes is not None:
             boxes = result.boxes.xyxy.cpu().numpy()
             for box in boxes:
                 x1, y1, x2, y2 = map(int, box)
                 if x2 > x1 and y2 > y1:
                     roi = frame[y1:y2, x1:x2]
                     if mask_mode == "blur":
                        roi = cv2.GaussianBlur(roi, (blur_intensity, blur_intensity), 0)
                        frame[y1:y2, x1:x2] = roi
                     else: # solid
                        frame[y1:y2, x1:x2] = (30, 30, 30)

        out.write(frame)
        
    cap.release()
    out.release()
    
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", temp_output, "-i", input_video_path, 
            "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-c:a", "aac", 
            final_output
        ], check=True, capture_output=True)
        return final_output
    except Exception as e:
        print(f"Ffmpeg error: {e}")
        return temp_output

with gr.Blocks(title="Smart Logo Masker") as demo:
    gr.Markdown("<h1 style='text-align: center;'>So Sánh Video Gốc & Đã Che Logo (YOLO Masking)</h1>")
    
    with gr.Row():
        vid1 = gr.Video(label="Video Gốc", elem_id="vid1")
        vid2 = gr.Video(label="Video Đã Che Logo", elem_id="vid2", interactive=False)
        
    with gr.Row():
        conf_slider = gr.Slider(0.1, 1.0, value=0.75, label="Độ tin cậy (Confidence)")
        iou_slider = gr.Slider(0.1, 1.0, value=0.45, label="Độ trùng lặp (IoU)")
        blur_slider = gr.Slider(1, 199, step=2, value=99, label="Cường độ làm mờ (Blur)")
        mask_mode = gr.Dropdown(["blur", "solid"], value="blur", label="Chế độ che")

    with gr.Row():
        play_btn = gr.Button("▶️ Phát cùng lúc", variant="primary")
        pause_btn = gr.Button("⏸️ Dừng cùng lúc", variant="stop")
        
    process_btn = gr.Button("🤖 Bắt Đầu Xử lý Che Logo (Phát Lại Sau Khi Xong)")
    
    play_btn.click(fn=None, inputs=None, outputs=None, js=js_play)
    pause_btn.click(fn=None, inputs=None, outputs=None, js=js_pause)
    
    process_btn.click(
        fn=process_video,
        inputs=[vid1, conf_slider, iou_slider, blur_slider, mask_mode],
        outputs=[vid2]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
