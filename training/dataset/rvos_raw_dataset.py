# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import logging
import os
from dataclasses import dataclass

from typing import List, Optional, Dict

import pandas as pd

import torch

from iopath.common.file_io import g_pathmgr

from omegaconf.listconfig import ListConfig

from training.dataset.rvos_segment_loader import (
    MultiplePNGSegmentLoader,
    PalettisedPNGSegmentLoader,
)
import json


@dataclass
class RVOSFrame:
    frame_idx: int
    frame_name: str
    image_path: str
    data: Optional[torch.Tensor] = None
    is_conditioning_only: Optional[bool] = False


@dataclass
class RVOSVideo:
    video_name: str
    video_id: int
    frames: List[RVOSFrame]
    meta_annotation: Dict

    def __len__(self):
        return len(self.frames)


class RVOSRawDataset:
    def __init__(self):
        pass

    def get_video(self, idx):
        raise NotImplementedError()


class PNGRawDataset(RVOSRawDataset):
    def __init__(
        self,
        img_folder,
        gt_folder,
        meta_folder=None,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        sample_rate=1,
        is_palette=True,
        single_object_mode=False,
        truncate_video=-1,
        frames_sampling_mult=False,
        used_object_ids=None,
        mult_ratio=1,
    ):
        self.img_folder = img_folder
        self.gt_folder = gt_folder
        self.meta_folder = meta_folder if meta_folder is not None else os.path.join(os.path.dirname(gt_folder), "Meta")
        self.sample_rate = sample_rate
        self.is_palette = is_palette
        self.single_object_mode = single_object_mode
        self.truncate_video = truncate_video
        self.used_object_ids = used_object_ids

        # Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = os.listdir(self.img_folder)

        # Read and process excluded files if provided
        if excluded_videos_list_txt is not None:
            with g_pathmgr.open(excluded_videos_list_txt, "r") as f:
                excluded_files = [os.path.splitext(line.strip())[0] for line in f]
        else:
            excluded_files = []

        # Check if it's not in excluded_files
        self.video_names = sorted(
            [video_name for video_name in subset if video_name not in excluded_files]
        )

        if self.single_object_mode:
            # single object mode
            self.video_names = sorted(
                [
                    os.path.join(video_name, obj)
                    for video_name in self.video_names
                    for obj in os.listdir(os.path.join(self.gt_folder, video_name))
                ]
            )

        if frames_sampling_mult:
            video_names_mult = []
            for video_name in self.video_names:
                num_frames = len(os.listdir(os.path.join(self.img_folder, video_name)))
                num_frames = int(num_frames * mult_ratio)
                video_names_mult.extend([video_name] * num_frames)
            self.video_names = video_names_mult

    def get_video(self, idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[idx]

        if self.single_object_mode:
            video_frame_root = os.path.join(
                self.img_folder, os.path.dirname(video_name)
            )
        else:
            video_frame_root = os.path.join(self.img_folder, video_name)

        video_mask_root = os.path.join(self.gt_folder, video_name)

        if self.is_palette:
            segment_loader = PalettisedPNGSegmentLoader(video_mask_root, self.used_object_ids)
        else:
            segment_loader = MultiplePNGSegmentLoader(
                video_mask_root, self.single_object_mode
            )

        all_frames = sorted(glob.glob(os.path.join(video_frame_root, "*.jpg")))
        if len(all_frames) == 0:
            all_frames = sorted(glob.glob(os.path.join(video_frame_root, "*.png")))
        if self.truncate_video > 0:
            all_frames = all_frames[: self.truncate_video]
        frames = []
        for _, fpath in enumerate(all_frames[:: self.sample_rate]):
            frame_name = os.path.basename(fpath).split(".")[0]
            fid = int(frame_name)
            frames.append(RVOSFrame(fid, frame_name=frame_name, image_path=fpath))

        meta_path = os.path.join(self.meta_folder, video_name + ".json")
        with g_pathmgr.open(meta_path, "r") as f:
            meta_annotation = json.load(f)

        video = RVOSVideo(video_name, idx, frames, meta_annotation)
        return video, segment_loader

    def __len__(self):
        return len(self.video_names)

if __name__ == '__main__':
    # test the PNGRawDataset
    img_folder = '/mnt/data0/haofeng/data/Ref-Endovis18/train/JPEGImages'
    gt_folder = '/mnt/data0/haofeng/data/Ref-Endovis18/train/Annotations'
    used_object_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 17]
    dataset = PNGRawDataset(img_folder, gt_folder, used_object_ids=used_object_ids)
    for idx in range(len(dataset)):
        video, segment_loader = dataset.get_video(idx)
        print(video.video_name)
