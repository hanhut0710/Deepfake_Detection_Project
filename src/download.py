import kagglehub
import os

original = "Original"
deepfakes = "Deepfakes"

path = kagglehub.dataset_download("xdxd003/ff-c23")

print("Download complete. Extracting files...")

deepfakes_path = os.path.join(path, "FaceForensics++", deepfakes)
original_path = os.path.join(path, "FaceForensics++", original)

print(f"Deepfakes path: {deepfakes_path}")
print(f"Original path: {original_path}")