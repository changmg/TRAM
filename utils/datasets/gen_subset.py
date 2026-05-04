import os
import random
import shutil

# num_samples   = 100000
num_samples   = 10 * 1000
original_root = '/mnt/haas/cmeng/dataset/imagenet/'
# subset_root   = '/mnt/haas/cmeng/dataset/imagenet_subset100k/'
subset_root   = '/mnt/haas/cmeng/dataset/imagenet_subset10k/'

# Gather all training image paths
train_dir = os.path.join(original_root, 'train')
all_paths = []
for class_name in os.listdir(train_dir):
    class_dir = os.path.join(train_dir, class_name)
    if not os.path.isdir(class_dir):
        continue
    for fname in os.listdir(class_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            all_paths.append(os.path.join(class_dir, fname))
print(f"Found {len(all_paths)} training images in {train_dir}")

# Randomly pick
subset_paths = random.sample(all_paths, num_samples)
print(f"Randomly selecting {len(subset_paths)} training images")

# Save list for reproducibility
# with open('imagenet_100k_train_paths.txt', 'w') as f:
with open('imagenet_10k_train_paths.txt', 'w') as f:
    for p in subset_paths:
        f.write(p + '\n')

# Copy into new tree
for src in subset_paths:
    # '/.../train/class_name/XYZ.JPEG'
    assert src.lower().endswith('.jpeg')
    _, _, split, class_name, filename = src.split(os.sep)[-5:]
    dst_dir = os.path.join(subset_root, split, class_name)
    print(f"Copying {src} to {dst_dir}/{filename}")
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(src, os.path.join(dst_dir, filename))