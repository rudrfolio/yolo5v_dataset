import os
import shutil
import random

# -----------------------------
# User Settings
# -----------------------------
SOURCE_DIR = "archive"              # original dataset
OUTPUT_DIR = "dataset_limited"      # final dataset folder
TARGET_SIZE_MB = 300              # target size in MB
# -----------------------------

TARGET_SIZE = TARGET_SIZE_MB * 1024 * 1024  # MB → bytes

# Step 1: Collect all images per class with sizes
class_images = {}
total_size = 0

for class_name in os.listdir(SOURCE_DIR):
    class_dir = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue

    images = []
    class_size = 0
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        if os.path.isfile(img_path):
            size = os.path.getsize(img_path)
            images.append((img_path, size))
            class_size += size

    if images:
        class_images[class_name] = images
        total_size += class_size

# Step 2: Calculate proportion of dataset for each class
selected = []
final_total_size = 0
for class_name, images in class_images.items():
    class_size = sum(size for _, size in images)
    class_ratio = class_size / total_size

    # allowed size for this class
    allowed_size = TARGET_SIZE * class_ratio

    # shuffle images
    random.shuffle(images)

    # pick images until allowed size reached
    class_selected = []
    size_so_far = 0
    for img_path, size in images:
        if size_so_far + size > allowed_size:
            break
        class_selected.append(img_path)
        size_so_far += size

    selected.append((class_name, class_selected, size_so_far))
    final_total_size += size_so_far

# Step 3: Copy selected images
for class_name, class_selected, _ in selected:
    dest_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(dest_dir, exist_ok=True)
    for img_path in class_selected:
        shutil.copy(img_path, os.path.join(dest_dir, os.path.basename(img_path)))

# Summary
print("✅ Final dataset created at:", OUTPUT_DIR)
print("Total size:", round(final_total_size / 1024 / 1024, 2), "MB")
for class_name, _, class_size in selected:
    print(f"  {class_name}: {round(class_size/1024/1024,2)} MB")
