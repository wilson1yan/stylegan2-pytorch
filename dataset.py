from io import BytesIO

import torch
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import h5py
import json
import os.path as osp
import warnings
import math
import pickle
from torchvision.datasets.video_utils import VideoClips
import torch.nn.functional as F


def preprocess(video, resolution):
    # video: THWC, {0, ..., 255}
    video = video.permute(0, 3, 1, 2).float() # TCHW
    t, c, h, w = video.shape

    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode='bilinear',
                          align_corners=False)
    
    scale = tuple([t / r for t, r in zip(target_size, (h, w))])

    # center crop
    t, c, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]

    video = 2 * (video / 255.) - 1
    return video


class SomethingSomething(Dataset):
    def __init__(self, path, transform, resolution=256):
        super().__init__()
        self.resolution = resolution

        self.root = path
        video_ids = json.load(open(osp.join(self.root, 'train_subset.json'), 'r'))
        to_exclude = json.load(open(osp.join(self.root, 'exclude.json'), 'r'))
        to_exclude = set(to_exclude)
        video_ids = list(filter(lambda vid: vid not in to_exclude, video_ids))

        files = [osp.join(self.root, '20bn-something-something-v2', f'{vid}.webm')
                 for vid in video_ids]
        
        warnings.filterwarnings('ignore')
        cache_file = osp.join(self.root, 'train_metadata_4.pkl')
        metadata = pickle.load(open(cache_file, 'rb'))
        clips = VideoClips(files, 1, _precomputed_metadata=metadata)
        self._clips = clips
    
    def __len__(self):
        return self._clips.num_clips()
    
    def __getitem__(self, idx):
        video = self._clips.get_clip(idx)[0]
        video = preprocess(video, self.resolution)
        return video[0]


class HDF5Dataset(Dataset):
    """ Generic dataset for data stored in h5py as uint8 numpy arrays.
    Reads videos in {0, ..., 255} and returns in range [-0.5, 0.5] """
    def __init__(self, path, transform, resolution=256):
        """
        Args:
            args.data_path: path to the pickled data file with the
                following format:
                {
                    'train_data': [B, H, W, 3] np.uint8,
                    'train_idx': [B], np.int64 (start indexes for each video)
                    'test_data': [B', H, W, 3] np.uint8,
                    'test_idx': [B'], np.int64
                }
            args.sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.resolution = resolution

        # read in data
        self.data_file = path
        self.data = h5py.File(self.data_file, 'r')
        self._images = self.data[f'train_data']

    def __getstate__(self):
        state = self.__dict__
        state['data'].close()
        state['data'] = None
        state['_images'] = None

        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.data = h5py.File(self.data_file, 'r')
        self._images = self.data[f'{self.prefix}_data']

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        img = self._images[idx] # HWC in {0, ..., 255}
        img = torch.FloatTensor(img) / 255.
        img = 2 * img - 1
        return img.movedim(-1, 0)


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img
