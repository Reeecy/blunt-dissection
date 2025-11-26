import os
import shutil
from sklearn.model_selection import train_test_split

BASE = "final_dataset"
OUT = "final_dataset_split"

# Create output folders
for split in ["train", "val"]:
    os.makedirs(os.path.join(OUT, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUT, split, "depth"), exist_ok=True)
    os.makedirs(os.path.join(OUT, split, "masks"), exist_ok=True)

# Collect all image files
image_dir = os.path.join(BASE, "images")
depth_dir = os.path.join(BASE, "depth")
mask_dir  = os.path.join(BASE, "masks")

all_images = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])

# 80% train / 20% val
train_files, val_files = train_test_split(all_images, test_size=0.2, random_state=42)

def copy_set(file_list, split):
    for fname in file_list:
        # Paths
        img_path   = os.path.join(image_dir, fname)
        depth_path = os.path.join(depth_dir, fname.replace(".png", ".npy"))
        mask_path  = os.path.join(mask_dir, fname)

        # Sanity checks
        if not os.path.exists(depth_path):
            print(f"WARNING: Missing depth → {depth_path}")
            continue
        if not os.path.exists(mask_path):
            print(f"WARNING: Missing mask → {mask_path}")
            continue

        # Copy
        shutil.copy(img_path,   os.path.join(OUT, split, "images", fname))
        shutil.copy(depth_path, os.path.join(OUT, split, "depth", fname.replace(".png", ".npy")))
        shutil.copy(mask_path,  os.path.join(OUT, split, "masks", fname))

# Copy files
copy_set(train_files, "train")
copy_set(val_files, "val")

print("✔ Finished creating final_dataset_split!")
print(f"   Train: {len(train_files)} images")
print(f"     Val: {len(val_files)} images")
