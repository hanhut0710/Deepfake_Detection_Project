import os
import re
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from collections import defaultdict


class DeepfakeSequenceDataset(Dataset):
    def __init__(self, root_dir, transform=None, seq_len=20):
        self.samples = []
        self.transform = transform
        self.seq_len = seq_len

        for label_name in ['real', 'fake']:
            label = 0 if label_name == 'real' else 1
            class_dir = os.path.join(root_dir, label_name)

            video_dict = defaultdict(list)

            for filename in os.listdir(class_dir):
                if filename.endswith('.jpg'):
                    match = re.match(r'(real|fake)_(\d+)_frame(\d+)\.jpg', filename)
                    if match:
                        video_id = match.group(2)
                        filepath = os.path.join(class_dir, filename)
                        video_dict[video_id].append(filepath)

            # sort frame theo thứ tự frame001 → frame020
            for video_id, frame_list in video_dict.items():
                frame_list = sorted(frame_list)

                if len(frame_list) == self.seq_len:
                    self.samples.append((frame_list, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]

        frames = []
        for path in frame_paths:
            image = Image.open(path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            frames.append(image)

        frames = torch.stack(frames)  # shape: (20, 3, 224, 224)

        return frames, label