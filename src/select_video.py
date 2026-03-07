import os
import random
import shutil

# đường dẫn dataset gốc
dataset_root = r"C:\Users\MSI VN\.cache\kagglehub\datasets\xdxd003\ff-c23\versions\1"

original_path = os.path.join(dataset_root, "FaceForensics++_C23", "Original")
deepfakes_path = os.path.join(dataset_root, "FaceForensics++_C23", "Deepfakes")

# dataset subset
subset_root = os.path.join(dataset_root, "subset")
subset_real = os.path.join(subset_root, "real")
subset_fake = os.path.join(subset_root, "fake")

os.makedirs(subset_real, exist_ok=True)
os.makedirs(subset_fake, exist_ok=True)

# lấy danh sách video
original_videos = os.listdir(original_path)
deepfake_videos = os.listdir(deepfakes_path)

# chọn ngẫu nhiên 100 video
real_sample = random.sample(original_videos, 100)
fake_sample = random.sample(deepfake_videos, 100)

print("Copying real videos...")

for video in real_sample:
    src = os.path.join(original_path, video)
    dst = os.path.join(subset_real, video)
    shutil.copy(src, dst)

print("Copying fake videos...")

for video in fake_sample:
    src = os.path.join(deepfakes_path, video)
    dst = os.path.join(subset_fake, video)
    shutil.copy(src, dst)

print("Done!")
print("Real videos:", len(os.listdir(subset_real)))
print("Fake videos:", len(os.listdir(subset_fake)))