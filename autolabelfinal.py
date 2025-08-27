import os
import glob
import shutil
import random
import yaml
import torch
import torchvision.transforms as T
from PIL import Image

# -------------------
# 1. Load pretrained DINO model
# -------------------
model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
model.eval()

# -------------------
# 2. Paths
# -------------------
DATASET_DIR = "dataset_limited"   # input dataset (folders = classes)
OUTPUT_DIR = "dataset_final"      # YOLO-ready output

# Create folders train/val with images and labels
for split in ["train", "val"]:
    for sub in ["images", "labels"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, sub), exist_ok=True)

# -------------------
# 3. Collect classes
# -------------------
classes = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
class_to_idx = {cls: i for i, cls in enumerate(classes)}
print("✅ Classes:", class_to_idx)

# -------------------
# 4. Transform for DINO
# -------------------
transform = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# -------------------
# 5. Collect and split images (80/20)
# -------------------
all_images = []
for cls in classes:
    cls_dir = os.path.join(DATASET_DIR, cls)
    for img_path in glob.glob(os.path.join(cls_dir, "*.*")):
        all_images.append((img_path, class_to_idx[cls]))

random.shuffle(all_images)
split_idx = int(0.8 * len(all_images))
train_imgs = all_images[:split_idx]
val_imgs = all_images[split_idx:]

# -------------------
# 6. Extract bbox using DINO and save
# -------------------
def process_and_save(images, split):
    for img_path, cls_id in images:
        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size

            # preprocess for DINO
            tensor = transform(img).unsqueeze(0)

            with torch.no_grad():
                attentions = model.get_last_selfattention(tensor)

            # Average attention
            attn = attentions[0, :, 0, 1:].mean(0).reshape(14, 14).cpu()
            attn_resized = T.Resize((h, w))(attn.unsqueeze(0).unsqueeze(0))[0, 0]

            # Threshold for bbox
            mask = attn_resized > attn_resized.mean()
            coords = torch.nonzero(mask)
            if coords.shape[0] == 0:
                continue

            y_min, x_min = coords.min(0).values
            y_max, x_max = coords.max(0).values

            # Normalize YOLO format
            x_center = ((x_min + x_max) / 2) / w
            y_center = ((y_min + y_max) / 2) / h
            bw = (x_max - x_min) / w
            bh = (y_max - y_min) / h

            # save image
            img_name = os.path.basename(img_path)
            out_img = os.path.join(OUTPUT_DIR, split, "images", img_name)
            shutil.copy(img_path, out_img)

            # save label
            label_name = os.path.splitext(img_name)[0] + ".txt"
            out_label = os.path.join(OUTPUT_DIR, split, "labels", label_name)
            with open(out_label, "w") as f:
                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

        except Exception as e:
            print(f"⚠️ Error processing {img_path}: {e}")

process_and_save(train_imgs, "train")
process_and_save(val_imgs, "val")

print("✅ Dataset prepared with DINO bounding boxes and 80/20 split!")

# -------------------
# 7. Write data.yaml
# -------------------
yaml_content = {
    "train": os.path.join(OUTPUT_DIR, "train/images"),
    "val": os.path.join(OUTPUT_DIR, "val/images"),
    "nc": len(classes),
    "names": classes
}

with open(os.path.join(OUTPUT_DIR, "data.yaml"), "w") as f:
    yaml.dump(yaml_content, f)

print("✅ data.yaml created with class names!")
