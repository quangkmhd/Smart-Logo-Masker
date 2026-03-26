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

def process_video(input_video_path):
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
    
    # Progress bar parameters
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Bắt đầu xử lý video có {total_frames} frames...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Sử dụng model.predict() tương tự predict.py để nhận diện
        results = model.predict(frame, conf=0.75, iou=0.45, verbose=False)
        result = results[0]
        
        # Che logo dựa trên kết quả Model
        if result.masks is not None:
            # Nếu model có segmentation mask, dùng mask để làm mờ chính xác hình dạng logo
            masks = result.masks.data.cpu().numpy()
            for mask in masks:
                mask_resized = cv2.resize(mask, (width, height))
                # Làm mờ khung hình
                blurred_frame = cv2.GaussianBlur(frame, (99, 99), 30)
                # Ghép khung hình mờ vào đúng phần diện tích mask
                mask_resized = np.expand_dims(mask_resized, axis=-1)
                frame = np.where(mask_resized > 0.5, blurred_frame, frame).astype(np.uint8)
        elif result.boxes is not None:
             # Fallback: Nếu không có mask mà chỉ có bounding box
             boxes = result.boxes.xyxy.cpu().numpy()
             for box in boxes:
                 x1, y1, x2, y2 = map(int, box)
                 roi = frame[y1:y2, x1:x2]
                 if roi.size > 0:
                     roi = cv2.GaussianBlur(roi, (99, 99), 30)
                     frame[y1:y2, x1:x2] = roi

        out.write(frame)
        
    cap.release()
    out.release()
    
    print("Hoàn tất chèn khung hình, đang chuyển đổi codec sang H.264 mp4 để có thể play mượt mà trên trình duyệt web...")
    try:
        # Convert video sang h264 và map cả âm thanh (nếu có) từ video gốc
        subprocess.run([
            "ffmpeg", "-y", 
            "-i", temp_output, 
            "-i", input_video_path, 
            "-map", "0:v", 
            "-map", "1:a?", 
            "-c:v", "libx264", 
            "-c:a", "aac", 
            final_output
        ], check=True, capture_output=True)
        return final_output
    except Exception as e:
        print(f"Khởi tạo ffmpeg thất bại: {e}")
        return temp_output

with gr.Blocks(title="Smart Logo Masker") as demo:
    gr.Markdown("<h1 style='text-align: center;'>So Sánh Video Gốc & Đã Che Logo (YOLO Masking)</h1>")
    
    with gr.Row():
        vid1 = gr.Video(label="Video Gốc", elem_id="vid1")
        vid2 = gr.Video(label="Video Đã Che Logo", elem_id="vid2", interactive=False)
        
    with gr.Row():
        play_btn = gr.Button("▶️ Phát cùng lúc", variant="primary")
        pause_btn = gr.Button("⏸️ Dừng cùng lúc", variant="stop")
        
    # Nút xử lý video từ file upload gốc sang video đã che
    process_btn = gr.Button("🤖 Bắt Đầu Xử lý Che Logo (Phát Lại Sau Khi Xong)")
    
    # Gắn sự kiện click bằng Javascript để can thiệp trực tiếp vào 2 thẻ Video HTML
    play_btn.click(fn=None, inputs=None, outputs=None, js=js_play)
    pause_btn.click(fn=None, inputs=None, outputs=None, js=js_pause)
    
    # Gắn sự kiện xử lý video qua backend Python, sau đó trả output ra vid2
    process_btn.click(
        fn=process_video,
        inputs=[vid1],
        outputs=[vid2]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
