import os
import random
import shutil

# đường dẫn dataset gốc
dataset_root = r"C:\Users\MSI VN\.cache\kagglehub\datasets\xdxd003\ff-c23\versions\1"

base = os.path.join(dataset_root, "FaceForensics++_C23")

original_path = os.path.join(base, "Original")

fake_paths = {
   "deepfakes": os.path.join(base, "Deepfakes"),
    "detection": os.path.join(base, "DeepFakeDetection"),
    "face2face": os.path.join(base, "Face2Face"),
    "faceswap": os.path.join(base, "FaceSwap"),
    "neural": os.path.join(base, "NeuralTextures"),
}

real_videos = 900
fake_videos = real_videos // len(fake_paths)

# dataset subset
subset_root = r"D:\TaiLieuHocTap\Deepfake_Detection_Project\src\data\video_test"
subset_real = os.path.join(subset_root, "real")
subset_fake = os.path.join(subset_root, "fake")

os.makedirs(subset_real, exist_ok=True)
os.makedirs(subset_fake, exist_ok=True)

# Real videos
original_videos = os.listdir(original_path)

real_sample = random.sample(original_videos, real_videos)


print("Copying real videos...")

for video in real_sample:
    src = os.path.join(original_path, video)
    dst = os.path.join(subset_real, video)
    shutil.copy(src, dst)

print("Copying fake videos...")

for key, path in fake_paths.items():

    vids = os.listdir(path)

    sample = random.sample(vids, fake_videos)

    for vid in sample:
        new_name = f"{key}_{vid}"
        shutil.copy(
            os.path.join(path, vid),
            os.path.join(subset_fake, new_name)
        )

print("Done!")
print("Real videos:", len(os.listdir(subset_real)))
print("Fake videos:", len(os.listdir(subset_fake)))