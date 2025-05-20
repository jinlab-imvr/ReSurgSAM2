# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from dataclasses import dataclass
from typing import List

from training.dataset.rvos_segment_loader import LazySegments
from training.dataset.rvos_expression import get_expression_and_category

MAX_RETRIES = 1000


@dataclass
class SampledFramesAndObjects:
    frames: List[int]
    object_ids: List[int]
    expressions: List[str]


class RVOSSampler:
    def __init__(self, sort_frames=True):
        # frames are ordered by frame id when sort_frames is True
        self.sort_frames = sort_frames

    def sample(self, video):
        raise NotImplementedError()


class RandomUniformSampler(RVOSSampler):
    def __init__(
        self,
        num_tracking_frames,
        max_num_objects,
        num_ref_frames=3,
        num_buffer_frames=0,
        reverse_time_prob=0.0,
        dataset_name='Ref-Endovis18'
    ):
        self.num_tracking_frames = num_tracking_frames
        self.num_ref_frames = num_ref_frames
        self.num_frames = self.num_tracking_frames + self.num_ref_frames
        # The buffer frames are used to sample the reference frames
        self.num_buffer_frames = num_buffer_frames
        self.max_num_objects = max_num_objects
        self.reverse_time_prob = reverse_time_prob
        self.dataset_name = dataset_name

    def sample(self, video, segment_loader, epoch=None):
        for retry in range(MAX_RETRIES):
            if len(video.frames) < self.num_frames:
                raise Exception(
                    f"Cannot sample {self.num_frames} frames from video {video.video_name} as it only has {len(video.frames)} annotated frames."
                )
            start = random.randrange(0, len(video.frames) - self.num_frames - self.num_buffer_frames + 1)
            frames = [(start + step) for step in range(self.num_frames + self.num_buffer_frames)]
            if random.uniform(0, 1) < self.reverse_time_prob:
                # Reverse time
                frames = frames[::-1]
            # [2， 3， 7]
            last_ref_frame_id = (self.num_buffer_frames+self.num_ref_frames-1)  # 4
            # the last frame for referring and 7 frames for tracking
            tracking_frames = frames[last_ref_frame_id:]
            # select 2 buffer frames
            ref_frames = random.sample(frames[:last_ref_frame_id], self.num_ref_frames-1)
            frames = ref_frames + tracking_frames
            frames = [video.frames[i] for i in frames]

            # Get first frame object ids
            visible_object_ids = []
            loaded_segms = segment_loader.load(frames[0].frame_idx)

            # first_appear_frame_name存储在referring最后一帧出现的object_id对应的frame_name
            first_appear_frame_name = {}
            if isinstance(loaded_segms, LazySegments):
                # LazySegments for SA1BRawDataset
                raise NotImplementedError()
                visible_object_ids = list(loaded_segms.keys())
            else:
                # To ensure that the last frame of num_ref_frames has a target
                for object_id, segment in segment_loader.load(
                        frames[self.num_ref_frames-1].frame_idx
                ).items():
                    if segment.sum():
                        visible_object_ids.append(object_id)
                        if object_id not in first_appear_frame_name:
                            first_appear_frame_name[object_id] = frames[self.num_ref_frames-1].frame_name

            # First frame needs to have at least a target to track
            if len(visible_object_ids) > 0:
                break
            if retry >= MAX_RETRIES - 1:
                raise Exception("No visible objects")

        object_ids = random.sample(
            visible_object_ids,
            min(len(visible_object_ids), self.max_num_objects),
        )
        # load the text data, video.meta_annotation, which is the first frame of the text data
        expressions = []
        for obj_id in object_ids:
            frame_name = first_appear_frame_name[obj_id]
            category = video.meta_annotation['info']['category'][str(obj_id)]
            action_id = video.meta_annotation['frames'][frame_name][str(obj_id)]['action']
            location_id = video.meta_annotation['frames'][frame_name][str(obj_id)]['location']
            is_unique = video.meta_annotation['frames'][frame_name][str(obj_id)]['unique']
            exp = get_expression_and_category(category, action_id, location_id, is_unique=is_unique, dataset_name=self.dataset_name)
            expressions.append(exp)

        return SampledFramesAndObjects(frames=frames, object_ids=object_ids, expressions=expressions)


class EvalSampler(RVOSSampler):
    """
    VOS Sampler for evaluation: sampling all the frames and all the objects in a video
    """

    def __init__(
        self,
    ):
        super().__init__()

    def sample(self, video, segment_loader, epoch=None):
        """
        Sampling all the frames and all the objects
        """
        if self.sort_frames:
            # ordered by frame id
            frames = sorted(video.frames, key=lambda x: x.frame_idx)
        else:
            # use the original order
            frames = video.frames
        object_ids = segment_loader.load(frames[0].frame_idx).keys()
        if len(object_ids) == 0:
            raise Exception("First frame of the video has no objects")

        return SampledFramesAndObjects(frames=frames, object_ids=object_ids)
