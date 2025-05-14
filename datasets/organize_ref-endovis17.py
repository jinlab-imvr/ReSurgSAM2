import os
import argparse
import warnings
from tqdm import tqdm
import cv2
import zipfile
import shutil


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description="Data organization routine Ref-EndoVis 2017 dataset")
    parser.add_argument(
        "--download_data_dir",
        type=str,
        default="./endovis17",
        help="path to the data",
    )
    parser.add_argument(
        "--ref_annotation_path",
        type=str,
        default="./endovis17/Ref-Endovis17.zip",
        help="path to the data",
    )
    parser.add_argument(
        "--unzip_dir",
        type=str,
        default="./endovis17_unzip",
        help="temporay directory to unzip the data"
    )
    parser.add_argument(
        "--target_dataset_root",
        type=str,
        default="./",
        help="path to the save the organized dataset",
    )
    return parser.parse_args()


def crop_image(image, h_start, w_start, h, w):
    image = image[h_start : h_start + h, w_start : w_start + w]
    return image


if __name__ == "__main__":
    args = parse_args()
    print("Called with args:")
    print(args)

    target_dataset_dir = os.path.join(args.target_dataset_root, 'Ref-Endovis17')

    h, w = 1024, 1280
    h_start, w_start = 28, 320

    dataset_list = [
        'instrument_1_4_testing.zip',
        'instrument_5_8_testing.zip',
        'instrument_1_4_training.zip',
        'instrument_5_8_training.zip',
        'instrument_9_10_testing.zip',
    ]

    # unzip the dataset
    if not os.path.exists(args.unzip_dir):
        os.makedirs(args.unzip_dir)
    for dataset_name in tqdm(dataset_list):
        dataset_path = os.path.join(args.download_data_dir, dataset_name)
        if not os.path.exists(dataset_path):
            print(f"{dataset_name} not found in {args.download_data_dir}")
            continue
        with zipfile.ZipFile(dataset_path, "r") as zip_ref:
            zip_ref.extractall(args.unzip_dir)

    # 1. get the left frame image, 2. crop the image, 3. rename and save
    instrument_dir_list = [d for d in os.listdir(args.unzip_dir) if d.startswith("instrument_")]
    train_seq_list = ['seq_1', 'seq_3', 'seq_4', 'seq_7', 'seq_8', 'seq_9', 'seq_10']
    valid_seq_list = ['seq_2', 'seq_5', 'seq_6']

    for instrument_dir in instrument_dir_list:
        # get the left frame image
        left_frames_dir = os.path.join(args.unzip_dir, instrument_dir, "left_frames")
        left_frames_list = os.listdir(left_frames_dir)
        left_frames_list.sort()

        # create the target directory
        target_name = f'seq_{instrument_dir.split("_")[-1]}'
        train_valid_split = 'train' if target_name in train_seq_list else 'valid'
        target_image_dir = os.path.join(target_dataset_dir, train_valid_split, "JPEGImages", target_name)
        if not os.path.exists(target_image_dir):
            os.makedirs(target_image_dir)

        # rename and save
        for left_frame in tqdm(left_frames_list):
            # get the image name
            source_image_name = left_frame
            source_image_path = os.path.join(left_frames_dir, source_image_name)
            # 'frame000.png' -> '00000.png'
            image_number = int(left_frame.replace('frame', '').split('.')[0])
            target_image_name = f"{image_number:05d}.png"
            target_image_path = os.path.join(target_image_dir, target_image_name)
            # read the image
            image = cv2.imread(source_image_path)
            # crop the image
            image = crop_image(image, h_start, w_start, h, w)
            # save the image
            cv2.imwrite(target_image_path, image)

    # unzip the annotation to the target_dataset_dir
    if not os.path.exists(args.ref_annotation_path):
        print(f"{args.ref_annotation_path} not found")
        exit(0)
    with zipfile.ZipFile(args.ref_annotation_path, "r") as zip_ref:
        # Unzip Ref-Endovis17
        zip_ref.extractall(args.target_dataset_root)

    # delete unzip_dir
    if os.path.exists(args.unzip_dir):
        shutil.rmtree(args.unzip_dir)
