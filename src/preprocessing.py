import os
import cv2
import random
import shutil
import torch
from sklearn.model_selection import train_test_split
from facenet_pytorch import MTCNN
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(BASE_DIR, "data", "video")
SPLIT_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset_faces")

FRAMES_PER_VIDEO = 20
IMG_SIZE = 224

device = "cuda" if torch.cuda.is_available() else "cpu"

detector = MTCNN(
    keep_all=False,
    device=device
)

#split dataset 

def split_dataset():

    for cls in ["real", "fake"]:

        path = os.path.join(INPUT_DIR, cls)
        videos = os.listdir(path)

        train, temp = train_test_split(videos, test_size=0.3, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)

        split_map = {
            "train_video": train,
            "val_video": val,
            "test_video": test
        }

        for split in split_map:

            save_dir = os.path.join(SPLIT_DIR, split, cls)
            os.makedirs(save_dir, exist_ok=True)

            for v in split_map[split]:

                src = os.path.join(path, v)
                dst = os.path.join(save_dir, v)

                shutil.copy(src, dst)

# vi du frames

def sample_frames(cap):

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    segment = max(total_frames // FRAMES_PER_VIDEO, 1)

    frame_ids = []

    for i in range(FRAMES_PER_VIDEO):

        start = i * segment
        end = min((i + 1) * segment, total_frames - 1)

        frame_ids.append(random.randint(start, end))

    return frame_ids

#face detection va crop

def extract_face(frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes, _ = detector.detect(rgb)

    if boxes is None:
        return None

    x1, y1, x2, y2 = map(int, boxes[0])

    h, w, _ = frame.shape

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    face = frame[y1:y2, x1:x2]

    if face.size == 0:
        return None

    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

    return face

# PROCESS VIDEO


def process_video(video_path, save_dir, cls, vid_id):

    cap = cv2.VideoCapture(video_path)

    frame_ids = sample_frames(cap)

    saved = 0

    for i, frame_id in enumerate(frame_ids):

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

        ret, frame = cap.read()

        if not ret:
            continue

        face = extract_face(frame)

        if face is None:
            continue

        name = f"{cls}_{vid_id}_frame{i+1:02d}.jpg"

        save_path = os.path.join(save_dir, name)

        cv2.imwrite(save_path, face)

        saved += 1

    cap.release()

    return saved

#  PREPROCESS DATASET


def preprocess():

    splits = ["train", "val", "test"]

    for split in splits:

        for cls in ["real", "fake"]:

            video_dir = os.path.join(SPLIT_DIR, f"{split}_video", cls)

            save_dir = os.path.join(OUTPUT_DIR, split)

            os.makedirs(save_dir, exist_ok=True)

            videos = os.listdir(video_dir)

            for idx, vid in enumerate(tqdm(videos, desc=f"{split}-{cls}")):

                video_path = os.path.join(video_dir, vid)

                vid_id = f"{idx:04d}"

                process_video(video_path, save_dir, cls, vid_id)


if __name__ == "__main__":

    split_dataset()

    preprocess()