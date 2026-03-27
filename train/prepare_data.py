import os
import json
import glob
import cv2
import shutil
import random

def prepare_yolo_dataset():
    RAW_DATA_DIR = "/home/quangnhvn34/dev/me/Smart-Logo-Masker/data_raw/Round_01"
    OUTPUT_DIR = "/home/quangnhvn34/dev/me/Smart-Logo-Masker/dataset"
    VAL_SPLIT = 0.1

    # Collect unique classes from subdirs based on expected names or discover them
    CLASSES = ["1xbet", "admiralbet", "eurobet", "melbet"]
    CLASS_2_ID = {c: i for i, c in enumerate(CLASSES)}

    # Create directories
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

    all_jsons = glob.glob(os.path.join(RAW_DATA_DIR, "*/*.json"))

    from collections import defaultdict
    class_groups = defaultdict(list)

    for json_path in all_jsons:
        img_path = json_path.replace('.json', '.jpg')
        if os.path.exists(img_path):
            folder_name = os.path.basename(os.path.dirname(json_path)).lower()
            assigned_class = "unknown"
            for c in CLASSES:
                if c in folder_name:
                    assigned_class = c
                    break
            class_groups[assigned_class].append((img_path, json_path))

    train_pairs = []
    val_pairs = []

    random.seed(42)
    for c, pairs in class_groups.items():
        random.shuffle(pairs)
        split_idx = int(len(pairs) * (1 - VAL_SPLIT))
        train_pairs.extend(pairs[:split_idx])
        val_pairs.extend(pairs[split_idx:])

    print(f"Found {len(train_pairs)} train and {len(val_pairs)} val image-json pairs across {len(class_groups)} categories.")

    def process_pairs(pairs, split):
        processed = 0
        for img_path, json_path in pairs:
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            lines = []
            for shape in data.get('shapes', []):
                label = shape['label']
                if label not in CLASS_2_ID:
                    # In case of minor variations (e.g. AdmiralBet instead of admiralbet)
                    label_lower = label.lower()
                    if label_lower in CLASS_2_ID:
                        label = label_lower
                    else:
                        continue
                class_id = CLASS_2_ID[label]
                points = shape['points']
                
                # Normalize points
                norm_points = []
                for x, y in points:
                    norm_points.append(str(round(max(0, min(x / w, 1)), 6)))
                    norm_points.append(str(round(max(0, min(y / h, 1)), 6)))
                
                # Format: class_id x1 y1 x2 y2 ...
                line = f"{class_id} " + " ".join(norm_points)
                lines.append(line)
            
            base_name = os.path.basename(img_path)
            out_img_path = os.path.join(OUTPUT_DIR, 'images', split, base_name)
            out_txt_path = os.path.join(OUTPUT_DIR, 'labels', split, base_name.replace('.jpg', '.txt'))
            
            shutil.copy(img_path, out_img_path)
            with open(out_txt_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(lines))
            processed += 1
        return processed

    print("Processing train split...")
    train_count = process_pairs(train_pairs, 'train')
    print("Processing val split...")
    val_count = process_pairs(val_pairs, 'val')

    print(f"Processed {train_count} train and {val_count} val files.")

    # Create data.yaml
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    yaml_content = f"""path: {OUTPUT_DIR}
train: images/train
val: images/val

names:
  0: 1xbet
  1: admiralbet
  2: eurobet
  3: melbet
"""
    with open(yaml_path, "w", encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"Dataset preparation complete! Configuration saved to {yaml_path}.")

if __name__ == "__main__":
    prepare_yolo_dataset()
