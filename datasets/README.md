# Datasets

Here we describe the steps for using the Endoscopic Vision 2017 [1] and 2018 [2] to construct Ref-Endovis17 and Ref-Endovis18.

## Notes on Dataset Correction

Minor annotation inconsistencies were found and corrected in the Ref-Endovis17 training set. In `seq_9`, the instrument labels were swapped and have been fixed as follows:  
`"2": "bipolar forceps", "4": "prograsp forceps"`.


## Downloading the data

1. Download the 2017 dataset from the challenge webpage [here](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/).
2. Download the 2018 dataset from the challenge webpage [here](https://endovissub2018-roboticscenesegmentation.grand-challenge.org/).
3. Download the Ref-Endovis17 [here](https://drive.google.com/file/d/1Fei1nPCfNNQX-co9x-Mh-A29ZjJL1kDe/view?usp=sharing) and Ref-Endovis18 [here](https://drive.google.com/file/d/1N_Xc0K3d7M_fnlRDx_gdyC6tfpl1aQWJ/view?usp=sharing).

The download dataset structure should look like:

```
project_root/
└── datasets/
    └── endovis17
    	└── instrument_1_4_testing.zip
        └── instrument_5_8_testing.zip
        └── instrument_1_4_training.zip
        └── instrument_5_8_training.zip
        └── instrument_9_10_testing.zip
        └── Ref-Endovis17.zip
    └── endovis18
    	└── miccai_challenge_2018_release_1.zip
        └── miccai_challenge_release_2.zip
        └── miccai_challenge_release_3.zip
        └── miccai_challenge_release_4.zip
        └── Ref-Endovis18.zip
```

## Organizing the data

In the working directory project_root/datasets, Run the following command:

```
python organize_ref-endovis17.py --download_data_dir ./endovis17 --ref_annotation_path ./endovis17/Ref-Endovis17.zip --unzip_dir./endovis17_unzip --target_dataset_root ./
```

```
python organize_ref-endovis18.py --download_data_dir ./endovis18 --ref_annotation_path ./endovis18/Ref-Endovis18.zip --unzip_dir./endovis17_unzip --target_dataset_root ./
```

The final datasets should be look like:

```
project_root/
└── datasets/
    └── endovis17
    └── endovis18
    └── Ref-Endovis17
        └── train
        	└── Annotations
        	└── JPEGImages
        	└── Meta
        └── valid
        	└── Annotations
        	└── JPEGImages
        	└── Meta
        	└── VOS
        	└── meta_expressions.json
    └── Ref-Endovis18
    	└── train
    		└── ...
    	└── valid
    		└── ...
```

## References

[1] Allan, M., Shvets, A., Kurmann, T., Zhang, Z., Duggal, R., Su, Y.H., , et al.: 2017 robotic instrument segmentation challenge. arXiv preprint arXiv:1902.06426 (2019) 

[2] Allan, M., Kondo, S., Bodenstedt, S., Leger, S., Kadkhodamohammadi, R., Luengo, I., Fuentes, F., Flouty, E., Mohammed, A., Pedersen, M., et al.: 2018 robotic scene segmentation challenge. arXiv preprint arXiv:2001.11190 (2020)

