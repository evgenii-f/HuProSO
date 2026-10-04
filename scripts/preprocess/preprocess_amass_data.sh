#!/bin/bash
# Paths come from config/paths.yaml (amass_raw, amass_rotations); see config/paths.example.yaml.

myArray=("TotalCapture" "EKUT" "ACCAD" "CMU" "KIT" "BMLmovi" "BMLrub" "EyesJapanDataset" "HDM05" "HumanEva" "MoSh" "SSM" "TCDHands" "Transitions" "WEIZMANN" "PosePrior" "SFU")

for dataset in ${myArray[@]}; do
    echo $dataset
    python ./scripts/preprocess/preprocess_data.py --dataset ${dataset}
done
