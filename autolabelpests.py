import os
import glob
import torch
import torchvision.transforms as T
from torchvision.ops import box_convert
from PIL import Image
import yaml

# -------------------
# 1. Download pretrained DINO model
# -------------------
model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
model.eval()

# -------------------
# 2. Paths setup
# -------------------
DATASET_DIR = "archive"   # change to your dataset root (with subfolders as classes)
OUTPUT_DIR = "output_dataset"
IMG_OUT = os.path.join(OUTPUT_DIR, "images")
LBL_OUT = os.path.join(OUTPUT_DIR, "labels")
os.makedirs(IMG_OUT, exist_ok=True)
os.makedirs(LBL_OUT, exist_ok=True)

# -------------------
# 3. Preprocess pipeline
# -------------------
transform = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225))
])

# -------------------
# 4. Collect classes from folder names
# -------------------
classes = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
class_to_idx = {cls: i for i, cls in enumerate(classes)}

# -------------------
# 5. Process images folder by folder
# -------------------
for cls in classes:
    class_folder = os.path.join(DATASET_DIR, cls)
    class_id = class_to_idx[cls]
    image_paths = glob.glob(os.path.join(class_folder, "*.jpg")) + glob.glob(os.path.join(class_folder, "*.png"))

    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            continue

        # Preprocess
        tensor = transform(img).unsqueeze(0)

        # Extract attention maps
        with torch.no_grad():
            attentions = model.get_last_selfattention(tensor)

        # Average attention
        attn = attentions[0, :, 0, 1:].mean(0).reshape(14, 14).cpu()
        attn_resized = T.Resize(img.size[::-1])(attn.unsqueeze(0).unsqueeze(0))[0, 0]

        # Threshold for bounding box
        mask = attn_resized > attn_resized.mean()
        coords = torch.nonzero(mask)
        if coords.shape[0] == 0:
            continue

        y_min, x_min = coords.min(0).values
        y_max, x_max = coords.max(0).values

        # Normalize to YOLO format
        x_center = ((x_min + x_max) / 2) / img.width
        y_center = ((y_min + y_max) / 2) / img.height
        w = (x_max - x_min) / img.width
        h = (y_max - y_min) / img.height

        # Save image to output/images
        img_name = os.path.basename(img_path)
        new_img_path = os.path.join(IMG_OUT, img_name)
        img.save(new_img_path)

        # Save YOLO label to output/labels
        label_path = os.path.splitext(img_name)[0] + ".txt"
        with open(os.path.join(LBL_OUT, label_path), "w") as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

print("✅ Dataset prepared with bounding boxes!")

# -------------------
# 6. Write data.yaml
# -------------------
yaml_content = {
    "train": IMG_OUT,
    "val": IMG_OUT,  # can split later
    "nc": len(classes),
    "names": classes
}

with open(os.path.join(OUTPUT_DIR, "data.yaml"), "w") as f:
    yaml.dump(yaml_content, f)

print("✅ data.yaml created with class names!")
