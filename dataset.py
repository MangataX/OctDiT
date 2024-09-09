import torch
import numpy as np
from torch.utils.data import Dataset
from jaxtyping import Int64, Float
from pathlib import Path
from typing import Literal, TypeAlias, List, Optional, Tuple
import random
from torch import Tensor

# taken from https://github.com/optas/latent_3d_points/blob/8e8f29f8124ed5fc59439e8551ba7ef7567c9a37/src/in_out.py
_ID2CATE = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
_CATE2ID = {v: k for k, v in _ID2CATE.items()}

ShapeNetCategory: TypeAlias = Literal[
    'airplane', 'bag', 'basket', 'bathtub', 'bed', 'bench', 'bottle',
    'bowl', 'bus', 'cabinet', 'can', 'camera', 'cap', 'car', 'chair',
    'clock', 'dishwasher', 'monitor', 'table', 'telephone', 'tin_can',
    'tower', 'train', 'keyboard', 'earphone', 'faucet', 'file', 'guitar',
    'helmet', 'jar', 'knife', 'lamp', 'laptop', 'speaker', 'mailbox',
    'microphone', 'microwave', 'motorcycle', 'mug', 'piano', 'pillow',
    'pistol', 'pot', 'printer', 'remote_control', 'rifle', 'rocket',
    'skateboard', 'sofa', 'stove', 'vessel', 'washer', 'cellphone',
    'birdhouse', 'bookshelf'
]


class ShapeNetDataset(Dataset[Tuple[
    Int64[Tensor, "1"],
    Float[Tensor, "num_points 3"],
]]):
    def __init__(
        self,
        categories: List[ShapeNetCategory],
        *,
        num_points: int,
        path: Path = Path('data') / 'ShapeNetCore.v2.PC15k',
        split: Literal['train', 'val', 'test'] = 'train',
        random_subsample: bool = False,
        all_points_mean: Optional[np.ndarray] = None,
        all_points_std: Optional[np.ndarray] = None,
        device: Optional[torch.device] = None
    ) -> None:
        assert num_points <= 15000
        self.split = split
        self.random_subsample = random_subsample
        self.num_points = num_points
        self.device = device
        # self.color = np.array([0.5, 0.5, 0.5])

        data = []
        for cate_idx, category in enumerate(categories):
            sub_path = path / _CATE2ID[category] / split
            assert sub_path.exists()
            for file in sub_path.glob("*.npy"):
                pcd = np.load(file)
                assert pcd.shape == (15000, 3)
                data.append((cate_idx, pcd))

        # Shuffle the index deterministically (based on the number of examples)
        random.Random(38383).shuffle(data)
        all_points = np.stack([pcd for cate_idx, pcd in data], axis=0)  # [N, 15000, 3]
        self.category_indices = torch.tensor([cate_idx for cate_idx, pcd in data]).long().to(device)
        if all_points_mean is not None and all_points_std is not None:  # using loaded dataset stats
            self.all_points_mean = all_points_mean
            self.all_points_std = all_points_std
        else:  # normalize across the dataset
            self.all_points_mean = all_points.reshape(-1, 3).mean(0).reshape(1, 1, 3)
            self.all_points_std = all_points.reshape(-1).std(0).reshape(1, 1, 1)

        # all_points = (all_points - self.all_points_mean) / self.all_points_std
        all_points = (all_points - all_points.min(0).min(0)) / (all_points.max(0).max(0) - all_points.min(0).min(0))
        self.train_points = torch.from_numpy(all_points).float().to(device) * 2 - 1

    def __len__(self) -> int:
        return self.category_indices.shape[0]

    def __getitem__(self, idx: int) -> Tuple[
        Int64[Tensor, "1"],
        Float[Tensor, "3 num_points"],
    ]:
        tr_out = self.train_points[idx]
        if self.random_subsample:
            tr_idxs = np.random.choice(tr_out.shape[0], self.num_points)
        else:
            tr_idxs = np.arange(self.num_points)

        # tr_out = torch.cat([tr_out, torch.tensor(self.color, device=self.device).expand_as(tr_out)], dim=1)
        return (
            self.category_indices[idx].to(self.device), 
            tr_out[tr_idxs, :].transpose(-1, -2).contiguous().to(self.device)
        )