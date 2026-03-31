import os
import csv
import cv2
import torch
import yt_dlp
import av
import queue
import threading
from tqdm import tqdm
from ultralytics import YOLO

def format_duration(seconds):
    if not seconds: return "N/A"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def process_video(video_path, output_root, model, device, current_total_frames, target_frames):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    video_output_dir = os.path.join(output_root, video_name)
    raw_dir = os.path.join(video_output_dir, "raw")
    pred_dir = os.path.join(video_output_dir, "pred")
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        # Lấy FPS của video (thường 24, 30 hoặc 60)
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        try:
            total_frames = stream.frames if stream.frames > 0 else int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
        except Exception:
            total_frames = 0
    except Exception as e:
        print(f"Cannot open {video_path} with pyav: {e}")
        return 0
    
    saved_count = 0
    frame_idx = 0
    
    # [TỐI ƯU 1] NHẢY KHUNG HÌNH (Skip frames)
    # Vì video 30 hình/giây chứa rất nhiều ảnh trùng lặp.
    # Lấy 3 frame mỗi giây
    skip_step = max(1, int(fps // 3))

    with tqdm(total=total_frames, desc=f"Predicting {video_name}", unit="frame") as pbar:
        try:
            for frame_av in container.decode(stream):
                frame_idx += 1
                
                # Check target an toàn
                if current_total_frames + saved_count >= target_frames:
                    break
                    
                # Nhảy cóc khung hình để tăng tốc đột phá
                if frame_idx % skip_step != 0:
                    pbar.update(1)
                    continue

                frame = frame_av.to_ndarray(format='bgr24')
                
                # Lựa chọn half=True trên GPU giúp tăng tốc predict (nếu máy có GPU)
                results = model(frame, imgsz=1280, device=device, verbose=False, half=(device.type == 'cuda'))
                
                for r in results:
                    has_boxes = r.boxes is not None and len(r.boxes) > 0
                    has_masks = hasattr(r, "masks") and r.masks is not None
                    
                    if not (has_boxes or has_masks):
                        continue
                        
                    saved_count += 1
                    
                    # [TỐI ƯU 2] VIẾT ẢNH CHẤT LƯỢNG CAO NHẤT
                    # Không resize raw, giữ chất lượng 100% khi decode
                    raw_path = os.path.join(raw_dir, f"{saved_count}.jpg")
                    cv2.imwrite(raw_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    
                    # Phục hồi lưu ảnh pred để bạn có thể xem model dự đoán ở đâu
                    im_array = r.plot()
                    pred_path = os.path.join(pred_dir, f"{saved_count}.jpg")
                    cv2.imwrite(pred_path, im_array)
                    
                pbar.set_postfix(saved=saved_count, total=current_total_frames + saved_count)
                pbar.update(1)
        except Exception as eval_err:
            print(f"Lỗi khi đọc frame trong {video_name}: {eval_err}")
            
    container.close()
    print(f"✅ Đã xử lý xong {video_name} | Thu được: {saved_count} frames")
    return saved_count

def downloader_task(entries, download_opts, stop_event, download_queue, processed_links):
    for i, entry in enumerate(entries, 1):
        if stop_event.is_set():
            break
            
        video_id = entry.get('id')
        if not video_id:
            continue
            
        link = f"https://www.youtube.com/watch?v={video_id}"
        if link in processed_links:
            continue
            
        print(f"\n[DOWNLOADER] Bắt đầu tải video {i}: {link}")
        
        try:
            with yt_dlp.YoutubeDL(download_opts) as ydl_dl:
                # Lấy thông tin sơ bộ để biết tên file sẽ tải
                info_dict = ydl_dl.extract_info(link, download=False)
                expected_file = ydl_dl.prepare_filename(info_dict)
                
                # Kiểm tra xem file đã tồn tại chưa (check cả mp4 và mkv)
                actual_file = None
                if os.path.exists(expected_file):
                    actual_file = expected_file
                else:
                    base_name = os.path.splitext(expected_file)[0]
                    for ext in ['.mp4', '.mkv', '.webm']:
                        if os.path.exists(base_name + ext):
                            actual_file = base_name + ext
                            break
                
                if actual_file:
                    print(f"   => [DOWNLOADER] File đã tồn tại: '{actual_file}'. Bỏ qua tải, đưa vào hàng chờ...")
                    downloaded_file = actual_file
                    actual_title = info_dict.get('title', 'Unknown')
                    duration = format_duration(info_dict.get('duration'))
                else:
                    # Nếu chưa có thì mới tải
                    info_dl = ydl_dl.extract_info(link, download=True)
                    actual_title = info_dl.get('title', 'Unknown')
                    duration = format_duration(info_dl.get('duration'))
                    downloaded_file = ydl_dl.prepare_filename(info_dl)
                    
                    if not os.path.exists(downloaded_file):
                        base_name = os.path.splitext(downloaded_file)[0]
                        if os.path.exists(base_name + ".mp4"):
                            downloaded_file = base_name + ".mp4"
                        elif os.path.exists(base_name + ".mkv"):
                            downloaded_file = base_name + ".mkv"
                        
                if os.path.exists(downloaded_file):
                    print(f"   => [DOWNLOADER] Đã tải xong: '{actual_title}'. Đang đưa vào hàng chờ...")
                    
                    # Chờ đưa vào hàng đợi (tối đa chứa 2 video tải sẵn)
                    while not stop_event.is_set():
                        try:
                            item = {
                                'index': i,
                                'file': downloaded_file,
                                'title': actual_title,
                                'link': link,
                                'duration': duration
                            }
                            # Block nếu hàng đợi đầy (đã tải sẵn đủ cữ)
                            download_queue.put(item, timeout=1)
                            print(f"   => [DOWNLOADER] Đã đưa '{actual_title}' vào hàng chờ dự trữ (GPU có thể lấy ngay).")
                            break
                        except queue.Full:
                            continue
                else:
                    print(f"   => [DOWNLOADER] Không tìm thấy file {downloaded_file} sau khi tải.")
                    
        except Exception as e:
            print(f"   => [DOWNLOADER] Lỗi tải video {link}: {e}")
            
    print("[DOWNLOADER] Đã xử lý hết danh sách tải hoặc có lệnh dừng.")
    # Đánh dấu đã duyệt hết danh sách video
    download_queue.put(None)


def main():
    target_frames = 1000
    current_total_frames = 0
    
    input_url = "https://www.youtube.com/@CNN/videos"
    
    downloads_folder = 'downloads'
    output_folder = 'out'
    csv_file = 'download_predict_log.csv'
    
    os.makedirs(downloads_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    
    model = YOLO('/home/quangnhvn34/data/fsoft/01_Raw/logo/data_background_3003/yolo11l-1280-vtv-04032026.pt')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Đọc log cũ để bỏ qua video đã xử lý và tiếp tục đếm số frame
    processed_links = set()
    if os.path.exists(csv_file):
        with open(csv_file, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 6:
                    processed_links.add(row[2]) # Cột Link
                    try:
                        current_total_frames = int(row[5]) # Cột Total Accumulated Frames
                    except ValueError:
                        pass
        print(f"Đã nạp {len(processed_links)} video từng chạy. Bắt đầu tiếp từ {current_total_frames}/{target_frames} frames.")
    else:
        with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["No", "Video Name", "Link", "Duration", "Frames Collected", "Total Accumulated Frames"])
            
    # Các tùy chọn cho việc lấy danh sách video
    ydl_opts_flat = {
        'extract_flat': True,
        'quiet': True,
        'no_warnings': True,
        'playlistend': 1000, # Giới hạn lấy 1000 video gần nhất
    }
    
    print(f"Đang lấy danh sách video từ {input_url}...")
    with yt_dlp.YoutubeDL(ydl_opts_flat) as ydl:
        info = ydl.extract_info(input_url, download=False)
        entries = info.get('entries', [])
        
    print(f"Tìm thấy {len(entries)} video trong danh sách (giới hạn 100 video).")
    
    # Các tùy chọn cho việc tải video
    download_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'merge_output_format': 'mp4',
        'outtmpl': f'{downloads_folder}/%(title)s.%(ext)s',
        'quiet': True,
        'no_warnings': True,
        
        # [TỐI ƯU TỐC ĐỘ DOWNLOAD]
        'concurrent_fragment_downloads': 10, # Bật tải đa luồng
        'http_chunk_size': 10485760, # Tải chunk 10MB (giúp lách giới hạn băng thông YouTube)
        
        # [LÁCH THROTTLE YOUTUBE MỚI NHẤT (2025)]
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'web']
            }
        }
    }
    
    # ================= LUỒNG TẢI NGẦM & HÀNG CHỜ =================
    # Hàng đợi tối đa 2 items -> Đảm bảo luôn tải trước 2 video, dự trữ 1, dự đoán 1
    download_queue = queue.Queue(maxsize=2)
    stop_event = threading.Event()
    
    # Bật luồng phụ chuyên tải video song song
    dl_thread = threading.Thread(
        target=downloader_task, 
        args=(entries, download_opts, stop_event, download_queue, processed_links)
    )
    dl_thread.start()
    # ==============================================================
    
    # LUỒNG CHÍNH LÀM NHIỆM VỤ DỰ ĐOÁN (PREDICT)
    while current_total_frames < target_frames:
        try:
            # Lấy video từ hàng chờ, chờ tối đa 1s để check event loop
            item = download_queue.get(timeout=1)
            
            if item is None:
                # None có nghĩa là luồng downloader đã duyệt hết danh sách tải
                break
                
            downloaded_file = item['file']
            actual_title = item['title']
            link = item['link']
            duration = item['duration']
            index = item['index']
            
            print(f"\n[PREDICTOR] Đang lấy video từ hàng chờ: {actual_title}")
            print(f"[PREDICTOR] Bắt đầu dự đoán trên thiết bị {device}...")
            
            saved_frames = process_video(
                downloaded_file, 
                output_folder, 
                model, 
                device, 
                current_total_frames, 
                target_frames
            )
            
            current_total_frames += saved_frames
            
            with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([index, actual_title, link, duration, saved_frames, current_total_frames])
                
            print(f"[PREDICTOR] Xong video! Đã thu được {saved_frames} frames.")
            print(f"[PREDICTOR] Tổng số khung hình đã duyệt: {current_total_frames}/{target_frames}")
            
            # Đánh dấu đã xong item
            download_queue.task_done()
            
            # Xóa video ngay sau khi xong để giải phóng ổ đĩa
            if os.path.exists(downloaded_file):
                try:
                    os.remove(downloaded_file)
                    print(f"[PREDICTOR] Đã xóa file video: {downloaded_file}")
                except Exception as e:
                    print(f"[PREDICTOR] Lỗi khi xóa video: {e}")
            
        except queue.Empty:
            # Hàng đợi rỗng -> Đang đợi downloader tải xong
            if not dl_thread.is_alive():
                print("\n[PREDICTOR] Luồng downloader chết bất thường, dừng dự đoán.")
                break
            continue
            
    if current_total_frames >= target_frames:
        print(f"\n🎉 Đã đạt đủ {target_frames} frames. Quá trình hoàn tất, dừng mọi hoạt động.")
    else:
        print(f"\n⚠️ Đã kết thúc nhưng chỉ thu được {current_total_frames}/{target_frames} frames.")

    # Ra hiệu lệnh dừng cho Downloader (nếu nó vẫn đang chạy)
    stop_event.set()
    
    # Xả hàng đợi để downloader không bị block và kết thúc mượt mà
    while not download_queue.empty():
        try:
            download_queue.get_nowait()
            download_queue.task_done()
        except:
            pass
            
    dl_thread.join()
    print("✅ Hoàn tất.")

if __name__ == '__main__':
    main()
