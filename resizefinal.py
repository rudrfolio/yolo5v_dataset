import os
import shutil
import random

# -----------------------------
# User Settings
# -----------------------------
SOURCE_DIR = "archive"              # original dataset
OUTPUT_DIR = "dataset_limited"      # final dataset folder
TARGET_SIZE_MB = 300                # target size in MB
# -----------------------------

TARGET_SIZE = TARGET_SIZE_MB * 1024 * 1024  # MB → bytes

# Step 1: Collect images per class
class_images = {}
for class_name in os.listdir(SOURCE_DIR):
    class_dir = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue
    images = []
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        if os.path.isfile(img_path):
            size = os.path.getsize(img_path)
            images.append((img_path, size))
    if images:
        class_images[class_name] = images

num_classes = len(class_images)

# Step 2: Distribute budget equally across classes
budget_per_class = TARGET_SIZE // num_classes

# Step 3: Select equal number of images per class (as close as possible)
selected = []
final_total_size = 0

# Find min number of images across all classes
min_images = min(len(imgs) for imgs in class_images.values())

for class_name, images in class_images.items():
    random.shuffle(images)

    class_selected = []
    size_so_far = 0
    for img_path, size in images:
        if size_so_far + size > budget_per_class:
            break
        class_selected.append(img_path)
        size_so_far += size

    # In case some classes have much fewer images, restrict others too
    if len(class_selected) > min_images:
        class_selected = class_selected[:min_images]
        size_so_far = sum(os.path.getsize(p) for p in class_selected)

    selected.append((class_name, class_selected, size_so_far))
    final_total_size += size_so_far

# Step 4: Copy selected images
for class_name, class_selected, _ in selected:
    dest_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(dest_dir, exist_ok=True)
    for img_path in class_selected:
        shutil.copy(img_path, os.path.join(dest_dir, os.path.basename(img_path)))

# Summary
print("✅ Final dataset created at:", OUTPUT_DIR)
print("Total size:", round(final_total_size / 1024 / 1024, 2), "MB")
for class_name, _, class_size in selected:
    print(f"  {class_name}: {round(class_size/1024/1024,2)} MB, {len(_)} images")
