# Smart Logo Masker 🛡️

Dự án này sử dụng YOLOv26 Segmentation để tự động phát hiện và che cho các logo cá độ (ví dụ: 1xbet, melbet, v.v.) trong video.

## 1. Cài đặt Dữ liệu (Setup Data)

Quy trình chuẩn bị dữ liệu bao gồm việc chuyển đổi các nhãn từ định dạng LabelMe (JSON) sang định dạng YOLO.

### Cấu trúc thư mục đầu vào:
Đặt dữ liệu thô vào thư mục `data_raw/Round_01/`. Mỗi thư mục con nên đại diện cho một class hoặc chứa tên class trong tên thư mục.
```text
data_raw/Round_01/
├── 1xbet/
│   ├── img1.jpg
│   ├── img1.json
├── admiralbet/
│   ├── img2.jpg
│   ├── img2.json
```

### Chạy script chuẩn bị:
Sử dụng script `prepare_data.py` để tự động phân chia tập train/val và tạo file `data.yaml`.
```bash
python3 prepare_data.py
```
Dữ liệu đã sẵn sàng sẽ nằm trong thư mục `dataset/`.

---

## 2. Huấn luyện Mô hình (Training)

Để bắt đầu huấn luyện, hãy chạy file `train_yolo.py`. Script này đã được cấu hình với các tham số tối ưu cho segmentation.

```bash
python3 train_yolo.py
```

### Các thông số quan trọng trong `train_yolo.py`:
- `model`: Model gốc (mặc định là `yolo26x-seg.pt`).
- `epochs`: Số vòng lặp huấn luyện (mặc định 300).
- `imgsz`: Kích thước ảnh đầu vào (640).
- `batch`: Số lượng ảnh trong một mẻ (giảm xuống nếu bị tràn RAM/VRAM).
- `patience`: Tự động dừng nếu model không cải thiện sau $N$ epoch (mặc định 50).

---

## 3. Dự đoán và Sử dụng (Inference)

Sau khi huấn luyện xong, bạn có thể sử dụng file `predict.py` để chạy trên dữ liệu mới.

```bash
python3 predict.py --source path/to/images --weights runs/segment/logo_masker_model/weights/best.pt --conf 0.75
```

---

## 4. Điều chỉnh Thông số (Parameter Tuning)

| Tham số | Ý nghĩa | Lời khuyên |
| :--- | :--- | :--- |
| `conf` | Ngưỡng tin cậy | Tăng lên (0.8+) nếu thấy nhiều logo mask sai. Giảm xuống nếu model bỏ sót logo. |
| `iou` | Ngưỡng NMS | Điều chỉnh nếu có nhiều mask chồng chéo cho cùng một logo. |
| `augmentations` | Tăng cường dữ liệu | Trong `train_yolo.py`, các thông số như `mosaic`, `mixup`, `copy_paste` giúp model học tốt hơn trong môi trường phức tạp. |
| `retina_masks` | Mask độ phân giải cao | Dùng `--retina_masks` trong `predict.py` để đường viền mask mịn hơn (nhưng chậm hơn). |

---

## 5. Lời khuyên về Dữ liệu Background (Background Data Advice) 💡

Để giảm thiểu tình trạng **Dương tính giả (False Positives)** - tức là model nhận nhầm vật thể khác là logo (ví dụ: một chiếc áo sọc hay biển báo giao thông), việc thêm dữ liệu background là cực kỳ quan trọng.

### Tại sao cần Background Data?
YOLO sẽ được học rằng "trong những bức ảnh này KHÔNG có logo nào cả". Điều này giúp model "tỉnh táo" hơn khi gặp các hình ảnh bình thường.

### Cách thêm Background Data:
1. **Dữ liệu**: Chọn các hình ảnh trông giống môi trường thực tế (sân vận động, đường phố, đám đông) nhưng **không chứa bất kỳ logo nào** mà bạn đang train.
2. **Quy trình nhãn**:
   - Đối với ảnh background, bạn chỉ cần tạo một file `.txt` trống (không chứa dòng nào) có cùng tên với ảnh trong thư mục `labels`.
   - Hoặc đơn giản là để ảnh vào thư mục `images` mà không có file `.txt` tương ứng trong `labels` (YOLO sẽ tự hiểu đó là background).
3. **Tỷ lệ**: Nên chiếm khoảng **10% - 15%** tổng số lượng dataset của bạn.
4. **Mẹo nhỏ**: Nếu model của bạn hay bị nhận nhầm một vật thể cố định nào đó (ví dụ: logo hãng giày Nike bị nhận nhầm là 1xbet), hãy chụp ảnh vật thể đó và đưa vào làm background images.

---

## Theo dõi quá trình huấn luyện
Bạn có thể sử dụng TensorBoard để theo dõi các chỉ số (mAP, loss):
```bash
tensorboard --logdir runs
```
