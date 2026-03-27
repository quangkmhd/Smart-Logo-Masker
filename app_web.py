import gradio as gr
import requests
import time
import os
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000/api/v1"

CUSTOM_CSS = """
body { background-color: #0f172a; color: #f8fafc; font-family: 'Inter', sans-serif; }
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }
.header { text-align: center; margin-bottom: 2rem; padding: 2rem 0; background: linear-gradient(90deg, #1e293b 0%, #334155 100%); border-radius: 12px; }
.header h1 { color: #38bdf8; margin: 0; font-weight: 800; font-size: 2.5rem; letter-spacing: -0.025em; }
.header p { color: #94a3b8; font-size: 1.1rem; }
.sidebar { background-color: #1e293b; padding: 1.5rem; border-radius: 12px; height: 100%; border: 1px solid #334155; }
.main-content { padding: 1.5rem; }
.btn-primary { background-color: #38bdf8 !important; border: none !important; color: #0f172a !important; font-weight: 600 !important; }
.btn-primary:hover { background-color: #7dd3fc !important; }
.video-preview { border-radius: 12px; overflow: hidden; border: 1px solid #334155; background-color: #020617; }
.status-box { padding: 1rem; border-radius: 8px; margin-top: 1rem; text-align: center; font-weight: 600; }
.status-pending { background-color: #1e293b; color: #94a3b8; }
.status-progress { background-color: #0c4a6e; color: #38bdf8; border: 1px solid #0369a1; }
.status-completed { background-color: #064e3b; color: #34d399; border: 1px solid #065f46; }
"""

def upload_and_process(video, conf, iou, blur_intensity, mask_mode):
    if not video:
        return None, "Vui lòng tải lên video."
    
    options = {
        "conf": conf,
        "iou": iou,
        "blur_intensity": blur_intensity,
        "mask_mode": mask_mode
    }
    
    try:
        with open(video, "rb") as f:
            files = {"file": (os.path.basename(video), f, "video/mp4")}
            data = {"options": json.dumps(options)}
            
            response = requests.post(f"{API_URL}/process", files=files, data=data)
            
            if response.status_code != 200:
                return None, f"Lỗi API: {response.text}"
            
            task_id = response.json()["task_id"]
            return task_id, "Tải lên thành công! Đang xử lý..."
            
    except Exception as e:
        return None, f"Lỗi kết nối: {str(e)}"

def check_status(task_id):
    if not task_id:
        return None, "Không có Task ID.", gr.update(visible=False)
    
    try:
        while True:
            response = requests.get(f"{API_URL}/status/{task_id}")
            if response.status_code != 200:
                return None, f"Lỗi kiểm tra trạng thái: {response.text}", gr.update(visible=False)
            
            data = response.json()
            status = data["status"]
            
            if status == "SUCCESS":
                download_res = requests.get(f"{API_URL}/download/{task_id}", stream=True)
                output_path = f"data/results/masked_ui_output_{task_id}.mp4"
                with open(output_path, "wb") as f:
                    for chunk in download_res.iter_content(chunk_size=8192):
                        f.write(chunk)
                return output_path, "Hoàn tất!", gr.update(visible=True, value=output_path)
            
            elif status == "FAILURE":
                return None, f"Xử lý thất bại: {data.get('result', {}).get('error', 'Unknown error')}", gr.update(visible=False)
            
            time.sleep(2)
            
    except Exception as e:
        return None, f"Lỗi: {str(e)}", gr.update(visible=False)

with gr.Blocks() as demo:
    with gr.Column(elem_classes="header"):
        gr.Markdown("# Smart Logo Masker 🛡️")
        gr.Markdown("Hệ thống che logo cá độ tự động - Chuyên nghiệp & Hiệu quả")

    with gr.Row():
        with gr.Column(scale=1, elem_classes="sidebar"):
            gr.Markdown("### ⚙️ Thông số xử lý")
            conf_slider = gr.Slider(0.1, 1.0, value=0.75, label="Độ tin cậy (Confidence)")
            iou_slider = gr.Slider(0.1, 1.0, value=0.45, label="Độ trùng lặp (IoU)")
            blur_slider = gr.Slider(1, 199, step=2, value=99, label="Cường độ làm mờ (Blur)")
            mask_mode = gr.Dropdown(["blur", "solid"], value="blur", label="Chế độ che")
            
            process_btn = gr.Button("🚀 Bắt đầu xử lý", variant="primary", elem_classes="btn-primary")
            status_text = gr.Markdown("Sẵn sàng.", elem_classes="status-box status-pending")
            task_id_state = gr.State()

        with gr.Column(scale=2, elem_classes="main-content"):
            with gr.Row():
                input_video = gr.Video(label="Video Gốc", elem_classes="video-preview")
                output_video = gr.Video(label="Kết quả", interactive=False, elem_classes="video-preview")

    # Event handlers
    process_btn.click(
        fn=upload_and_process,
        inputs=[input_video, conf_slider, iou_slider, blur_slider, mask_mode],
        outputs=[task_id_state, status_text]
    ).then(
        fn=check_status,
        inputs=[task_id_state],
        outputs=[output_video, status_text, output_video]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        share=False,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft()
    )
