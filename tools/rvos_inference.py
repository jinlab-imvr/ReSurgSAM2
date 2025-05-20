# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from collections import defaultdict

import numpy as np
import torch
import json
from PIL import Image
from sympy import evaluate

from sam2.build_sam import build_sam2_video_predictor
from sav_dataset.utils.my_sav_benchmark import benchmark

os.environ["TOKENIZERS_PARALLELISM"] = "false"

Endovis_PALETTE = b'\x00\x00\x00\xec_g\xf9\x91W\xfa\xc8c\x99\xc7\x94b\xb3\xb2f\x99\xcc\xc5\x94\xc5\x00\xff\x00\x00\xff\xff}\xff\x0c\xff7\x00\x187}\xbb\x9b\x19\x00\xff}\xff\xff}{\x0f\xaf|\x9b\x05\x0c\xff\x8d\x13\x13\x13\x14\x14\x14\x15\x15\x15\x16\x16\x16\x17\x17\x17\x18\x18\x18\x19\x19\x19\x1a\x1a\x1a\x1b\x1b\x1b\x1c\x1c\x1c\x1d\x1d\x1d\x1e\x1e\x1e\x1f\x1f\x1f   !!!"""###$$$%%%&&&\'\'\'((()))***+++,,,---...///000111222333444555666777888999:::;;;<<<===>>>???@@@AAABBBCCCDDDEEEFFFGGGHHHIIIJJJKKKLLLMMMNNNOOOPPPQQQRRRSSSTTTUUUVVVWWWXXXYYYZZZ[[[\\\\\\]]]^^^___```aaabbbcccdddeeefffggghhhiiijjjkkklllmmmnnnooopppqqqrrrssstttuuuvvvwwwxxxyyyzzz{{{|||}}}~~~\x7f\x7f\x7f\x80\x80\x80\x81\x81\x81\x82\x82\x82\x83\x83\x83\x84\x84\x84\x85\x85\x85\x86\x86\x86\x87\x87\x87\x88\x88\x88\x89\x89\x89\x8a\x8a\x8a\x8b\x8b\x8b\x8c\x8c\x8c\x8d\x8d\x8d\x8e\x8e\x8e\x8f\x8f\x8f\x90\x90\x90\x91\x91\x91\x92\x92\x92\x93\x93\x93\x94\x94\x94\x95\x95\x95\x96\x96\x96\x97\x97\x97\x98\x98\x98\x99\x99\x99\x9a\x9a\x9a\x9b\x9b\x9b\x9c\x9c\x9c\x9d\x9d\x9d\x9e\x9e\x9e\x9f\x9f\x9f\xa0\xa0\xa0\xa1\xa1\xa1\xa2\xa2\xa2\xa3\xa3\xa3\xa4\xa4\xa4\xa5\xa5\xa5\xa6\xa6\xa6\xa7\xa7\xa7\xa8\xa8\xa8\xa9\xa9\xa9\xaa\xaa\xaa\xab\xab\xab\xac\xac\xac\xad\xad\xad\xae\xae\xae\xaf\xaf\xaf\xb0\xb0\xb0\xb1\xb1\xb1\xb2\xb2\xb2\xb3\xb3\xb3\xb4\xb4\xb4\xb5\xb5\xb5\xb6\xb6\xb6\xb7\xb7\xb7\xb8\xb8\xb8\xb9\xb9\xb9\xba\xba\xba\xbb\xbb\xbb\xbc\xbc\xbc\xbd\xbd\xbd\xbe\xbe\xbe\xbf\xbf\xbf\xc0\xc0\xc0\xc1\xc1\xc1\xc2\xc2\xc2\xc3\xc3\xc3\xc4\xc4\xc4\xc5\xc5\xc5\xc6\xc6\xc6\xc7\xc7\xc7\xc8\xc8\xc8\xc9\xc9\xc9\xca\xca\xca\xcb\xcb\xcb\xcc\xcc\xcc\xcd\xcd\xcd\xce\xce\xce\xcf\xcf\xcf\xd0\xd0\xd0\xd1\xd1\xd1\xd2\xd2\xd2\xd3\xd3\xd3\xd4\xd4\xd4\xd5\xd5\xd5\xd6\xd6\xd6\xd7\xd7\xd7\xd8\xd8\xd8\xd9\xd9\xd9\xda\xda\xda\xdb\xdb\xdb\xdc\xdc\xdc\xdd\xdd\xdd\xde\xde\xde\xdf\xdf\xdf\xe0\xe0\xe0\xe1\xe1\xe1\xe2\xe2\xe2\xe3\xe3\xe3\xe4\xe4\xe4\xe5\xe5\xe5\xe6\xe6\xe6\xe7\xe7\xe7\xe8\xe8\xe8\xe9\xe9\xe9\xea\xea\xea\xeb\xeb\xeb\xec\xec\xec\xed\xed\xed\xee\xee\xee\xef\xef\xef\xf0\xf0\xf0\xf1\xf1\xf1\xf2\xf2\xf2\xf3\xf3\xf3\xf4\xf4\xf4\xf5\xf5\xf5\xf6\xf6\xf6\xf7\xf7\xf7\xf8\xf8\xf8\xf9\xf9\xf9\xfa\xfa\xfa\xfb\xfb\xfb\xfc\xfc\xfc\xfd\xfd\xfd\xfe\xfe\xfe\xff\xff\xff'


def load_ann_png(path):
    """Load a PNG file as a mask and its palette."""
    mask = Image.open(path)
    palette = mask.getpalette()
    mask = np.array(mask).astype(np.uint8)
    return mask, palette


def save_ann_png(path, mask, palette):
    """Save a mask as a PNG file with the given palette."""
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    output_mask = Image.fromarray(mask)
    output_mask.putpalette(palette)
    output_mask.save(path)


def get_per_obj_mask(mask):
    """Split a mask into per-object masks."""
    object_ids = np.unique(mask)
    object_ids = object_ids[object_ids > 0].tolist()
    per_obj_mask = {object_id: (mask == object_id) for object_id in object_ids}
    return per_obj_mask


def put_per_obj_mask(per_obj_mask, height, width):
    """Combine per-object masks into a single mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    object_ids = sorted(per_obj_mask)[::-1]
    for object_id in object_ids:
        object_mask = per_obj_mask[object_id]
        object_mask = object_mask.reshape(height, width)
        mask[object_mask] = object_id
    return mask


def load_masks_from_dir(
    input_mask_dir, video_name, frame_name, per_obj_png_file, allow_missing=False
):
    """Load masks from a directory as a dict of per-object masks."""
    if not per_obj_png_file:
        input_mask_path = os.path.join(input_mask_dir, video_name, f"{frame_name}.png")
        if not os.path.exists(input_mask_path):
            pass
        if allow_missing and not os.path.exists(input_mask_path):
            return {}, None
        input_mask, input_palette = load_ann_png(input_mask_path)
        per_obj_input_mask = get_per_obj_mask(input_mask)
    else:
        per_obj_input_mask = {}
        input_palette = None
        # each object is a directory in "{object_id:%03d}" format
        for object_name in os.listdir(os.path.join(input_mask_dir, video_name)):
            object_id = int(object_name)
            input_mask_path = os.path.join(
                input_mask_dir, video_name, object_name, f"{frame_name}.png"
            )
            if allow_missing and not os.path.exists(input_mask_path):
                continue
            input_mask, input_palette = load_ann_png(input_mask_path)
            per_obj_input_mask[object_id] = input_mask > 0

    return per_obj_input_mask, input_palette


def load_text_from_json(
    input_expression_json, video_name
):
    per_exp_input_text = {}
    per_exp_obj_id = {}
    per_exp_first_frame_id = {}
    input_palette = None
    with open(input_expression_json, 'r') as f:
        meta_expressions = json.load(f)
    videos = meta_expressions['videos']
    this_video = videos[video_name]
    expressions = this_video['expressions']
    frames = this_video['frames']
    first_frame = this_video['first_frame']
    for exp_id, exp_item in expressions.items():
        per_exp_input_text[exp_id] = exp_item['exp']
        per_exp_obj_id[exp_id] = int(exp_item['obj_id'])
    for exp_id, frame_item in first_frame.items():
        per_exp_first_frame_id[exp_id] = int(frame_item)

    return per_exp_input_text, per_exp_obj_id, per_exp_first_frame_id, input_palette


def save_masks_to_dir(
    output_mask_dir,
    video_name,
    frame_name,
    per_obj_output_mask,
    height,
    width,
    output_per_obj_png_file,
    output_palette,
):
    """Save masks to a directory as PNG files."""
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    if not output_per_obj_png_file:
        os.makedirs(os.path.join(output_mask_dir, video_name, 'all'), exist_ok=True)
        output_mask = put_per_obj_mask(per_obj_output_mask, height, width)
        output_mask_path = os.path.join(
            output_mask_dir, video_name, 'all', f"{frame_name}.png"
        )
        save_ann_png(output_mask_path, output_mask, output_palette)
    else:
        for object_id, object_mask in per_obj_output_mask.items():
            object_name = f"{object_id:03d}"
            os.makedirs(
                os.path.join(output_mask_dir, video_name, object_name),
                exist_ok=True,
            )
            output_mask = object_mask.reshape(height, width).astype(np.uint8)
            output_mask[output_mask>0] = object_id
            output_mask_path = os.path.join(
                output_mask_dir, video_name, object_name, f"{frame_name}.png"
            )
            save_ann_png(output_mask_path, output_mask, output_palette)


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def rvos_inference(
    predictor,
    base_video_dir,
    input_expression_json,
    output_mask_dir,
    video_name,
    score_thresh=0.0,
    read_frame_interval=1,
    save_frame_interval=1,
    ref_dataset=True,
):
    """
    Run VOS inference on a single video with the given predictor.

    Unlike `vos_inference`, this function run inference separately for each object
    in a video, which could be applied to datasets like LVOS or YouTube-VOS that
    don't have all objects to track appearing in the first frame (i.e. some objects
    might appear only later in the video).
    """
    # load the video frames and initialize the inference state on this video
    assert ref_dataset, "This function is only for reference datasets"
    video_dir = os.path.join(base_video_dir, video_name)
    all_frame_names = [
        os.path.splitext(p)[0]
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", '.png']
    ]
    # only process frames with frame_interval
    frame_names = [p for p in all_frame_names if int(os.path.splitext(p)[0]) % read_frame_interval == 0]
    # save_pred_freq = len(all_frame_names) // len(frame_names)
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    inference_state = predictor.init_state(
        video_path=video_dir, async_loading_frames=True, frame_interval=read_frame_interval
    )
    height = inference_state["video_height"]
    width = inference_state["video_width"]

    inputs_per_object = defaultdict(dict)
    per_exp_input_text, per_exp_obj_id, per_exp_first_frame_id, input_palette = load_text_from_json(
        input_expression_json=input_expression_json, video_name=video_name
    )
    for i, exp_id in enumerate(per_exp_input_text):
        object_id = per_exp_obj_id[exp_id]
        inputs_per_object[object_id][0] = per_exp_input_text[exp_id]
    output_palette = input_palette or Endovis_PALETTE

    # step 1: run inference separately for the object appearing in the latter frame
    # for object_id in latter_frame_object_ids:
    object_ids = sorted(inputs_per_object)
    output_scores_per_object = defaultdict(dict)
    predictor.reset_state(inference_state)
    for object_id in object_ids:
        predictor.add_new_text(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=object_id,
            text=inputs_per_object[object_id][0],
        )
    # run propagation throughout the video and collect the results in a dict
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=0,
        reverse=False,
    ):
        # obj_scores = out_mask_logits.cpu().numpy()
        obj_scores = out_mask_logits
        for i, out_obj_id in enumerate(out_obj_ids):
            output_scores_per_object[out_obj_id][out_frame_idx] = obj_scores[i: i + 1]

    video_segments = {}
    for out_frame_idx in range(len(frame_names)):
        frame_time = int(frame_names[out_frame_idx])
        if frame_time % save_frame_interval != 0:    # save_frame_interval，控制保存频率
            continue
        video_segments[out_frame_idx] = {}
        for object_id in object_ids:
            if output_scores_per_object[object_id].keys().__contains__(out_frame_idx):
                video_segments[out_frame_idx][object_id] = (
                    (output_scores_per_object[object_id][out_frame_idx] > score_thresh).cpu().numpy()
                )

    # step 2: save the output masks as per-object PNG files
    for out_frame_idx, per_obj_output_mask in video_segments.items():
        save_masks_to_dir(
            output_mask_dir=output_mask_dir,
            video_name=video_name,
            per_obj_output_mask=per_obj_output_mask,
            frame_name=frame_names[out_frame_idx],
            output_per_obj_png_file=True,
            height=height,
            width=width,
            output_palette=output_palette,
        )

    # step 3: save the output masks as a single PNG file
    # post-processing: consolidate the per-object scores into per-frame masks
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for frame_idx in range(len(frame_names)):
        frame_time = int(frame_names[frame_idx])
        if frame_time % save_frame_interval != 0:
            continue
        scores = torch.full(
            size=(len(object_ids), 1, height, width),
            fill_value=-1024.0,
            dtype=torch.float32,
            device='cuda',
        )
        for i, object_id in enumerate(object_ids):
            if frame_idx in output_scores_per_object[object_id]:
                scores[i] = output_scores_per_object[object_id][frame_idx]

        scores = predictor._apply_non_overlapping_constraints(scores)
        per_obj_output_mask = {
            object_id: (scores[i] > score_thresh).cpu().numpy()
            for i, object_id in enumerate(object_ids)
        }
        video_segments[frame_idx] = per_obj_output_mask

    # write the output masks as palette PNG files to output_mask_dir
    for frame_idx, per_obj_output_mask in video_segments.items():
        save_masks_to_dir(
            output_mask_dir=output_mask_dir,
            video_name=video_name,
            frame_name=frame_names[frame_idx],
            per_obj_output_mask=per_obj_output_mask,
            height=height,
            width=width,
            output_per_obj_png_file=False,
            output_palette=output_palette,
        )


def benchmark_ref18(args, gt_root, output_mask_dir):
    evaluate_object_id_groups = {
        'instrument': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'tissue': [11, 12, 17]
    }
    for object_id_group_name, evaluate_object_id_list in evaluate_object_id_groups.items():
        print(f'Running evaluation for {object_id_group_name}')
        benchmark(
            [gt_root],
            [output_mask_dir],
            strict=False,
            num_processes=args.num_processes,
            verbose=not args.quiet,
            evaluate_object_id_list=evaluate_object_id_list,
            ref_dataset=True
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sam2_cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_s_rvos.yaml",
        help="SAM 2 model configuration file",
    )
    parser.add_argument(
        "--training_config_file",
        type=str,
        default=""
    )
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default="./checkpoints/sam2.1_hiera_b+.pt",
        help="path to the SAM 2 model checkpoint",
    )

    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="directory containing videos (as JPEG files) to run VOS prediction on",
    )
    parser.add_argument(
        "--output_mask_dir",
        type=str,
        required=True,
        help="directory to save the output masks (as PNG files)",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.0,
        help="threshold for the output mask logits (default: 0.0)",
    )
    parser.add_argument(
        "--apply_postprocessing",
        action="store_true",
        help="whether to apply postprocessing (e.g. hole-filling) to the output masks "
        "(we don't apply such post-processing in the SAM 2 model evaluation)",
    )
    parser.add_argument(
        "--apply_long_term_memory",
        action="store_true",
        help="whether to apply long term memory",
    )
    parser.add_argument(
        "--apply_credible_initial_frame",
        type=int, default=1,
    )
    parser.add_argument(
        "--num_cand_to_cond_frame",
        type=int,
        default=1,
    )
    # num_cand_to_cond_frame
    parser.add_argument(
        "--num_cifs_candidate_frame",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--num_long_mem_frame",
        type=int,
        default=3,
    )

    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--read_frame_interval", type=int, default=1)
    parser.add_argument("--save_frame_interval", type=int, default=1)
    # ----
    parser.add_argument(
        "-n", "--num_processes", default=16, type=int, help="Number of concurrent processes"
    )
    parser.add_argument(
        "-s",
        "--strict",
        help="Make sure every video in the gt_root folder has a corresponding video in the prediction",
        action="store_true",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        help="Quietly run evaluation without printing the information out",
        action="store_true",
    )
    args = parser.parse_args()

    base_video_dir = os.path.join(args.dataset_root, "JPEGImages")
    gt_root = os.path.join(args.dataset_root, "Annotations")
    input_expression_json = os.path.join(args.dataset_root, "meta_expressions.json")

    torch.cuda.set_device(args.gpu_id)
    print(f"Using GPU {args.gpu_id}")
    print('Warning: only support evaluating one object per sequence and saving in object directories.')

    # print('Testing with first frame prompt file:', args.first_frame_prompt_file)

    predictor = build_sam2_video_predictor(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        apply_postprocessing=args.apply_postprocessing,
        hydra_overrides_extra=[],
        strict_loading=False,
        training_config_file=args.training_config_file,
        apply_long_term_memory=args.apply_long_term_memory,
        apply_credible_initial_frame=args.apply_credible_initial_frame,
        num_cand_to_cond_frame=args.num_cand_to_cond_frame,
        num_cifs_candidate_frame=args.num_cifs_candidate_frame,
        num_long_mem_frame=args.num_long_mem_frame,
    )

    video_names = [
        p
        for p in os.listdir(base_video_dir)
        if os.path.isdir(os.path.join(base_video_dir, p))
    ]
    #
    # we first run every object separately and then combine them
    for n_video, video_name in enumerate(video_names):
        print(f"\n{n_video + 1}/{len(video_names)} - running on {video_name}")
        rvos_inference(
            predictor=predictor,
            base_video_dir=base_video_dir,
            input_expression_json=input_expression_json,
            output_mask_dir=args.output_mask_dir,
            video_name=video_name,
            score_thresh=args.score_thresh,
            read_frame_interval=args.read_frame_interval,
            save_frame_interval=args.save_frame_interval,
        )

    print(
        f"completed VOS prediction on {len(video_names)} videos -- "
        f"output masks saved to {args.output_mask_dir}"
    )

    if '18' in gt_root:
        benchmark_ref18(args, gt_root, args.output_mask_dir)
        return
    benchmark(
        [gt_root],
        [args.output_mask_dir],
        args.strict,
        args.num_processes,
        verbose=not args.quiet,
        ref_dataset=True
    )


if __name__ == "__main__":
    main()
