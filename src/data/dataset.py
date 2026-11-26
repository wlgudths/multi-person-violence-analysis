from pathlib import Path
from torch.utils.data import Dataset

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mpeg", ".mov", ".mkv"}


class GcnDataset(Dataset):
    def __init__(self, cfg, test=True):
        self.cfg = cfg

        if test:
            self.data_path = Path(cfg["data"]["test_dataset"])
        else:
            self.data_path = Path(cfg["data"]["exp_dataset"])

        self.target_fps = cfg["data"]["target_fps"]
        self.target_long = cfg["data"]["target_long"]

        if self.data_path.is_file():
            self.videos = [self.data_path]

        elif self.data_path.is_dir():
            self.videos = sorted([
                path
                for path in self.data_path.rglob("*")
                if path.suffix.lower() in VIDEO_EXTENSIONS
            ])

        else:
            raise FileNotFoundError(f"Dataset path not found: {self.data_path}")

        if len(self.videos) == 0:
            raise RuntimeError(f"No video files found: {self.data_path}")
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        video_path = self.videos[idx]

        return {
            "video_id": video_path.stem,
            "video_path": video_path,
        }

