#!/bin/bash

mkdir -p RIMES
cd RIMES


trainingsnippets_icdar_rar='1PDAbB96LFkD29QQIJkl1b1p4w3Gl8nuc'
validationsnippets_icdar_tar_gz='16FF0uMvC1aKbzPCFm9HzJUAK8OUoL4IM'
data_test_tar_gz='1rnnOmrD6J4bTYWJrUpYm6an76nTWgnci'

groundtruth_training_icdar2011_txt='177XScNrWnOK3L5to_qFlA6S5AIO-cWuO'
ground_truth_validation_icdar2011_txt='1FRnop_CeOxYt1yi_XQDS0n92JoEMw_dB'
grount_truth_test_icdar2011_txt='1m65IbSdLqYDeuig7NTeJtRXDsvpGERg9'

function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}


function gdrive_download_small () {
  wget --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O $2
}

echo 'Download training images...'
gdrive_download $trainingsnippets_icdar_rar 'trainingsnippets_icdar.rar'

echo 'Download training labels...'
gdrive_download_small $groundtruth_training_icdar2011_txt 'groundtruth_training_icdar2011.txt'

echo 'Download validation images...'
gdrive_download $validationsnippets_icdar_tar_gz 'validationsnippets_icdar.tar.gz'

echo 'Download validation labels...'
gdrive_download_small $ground_truth_validation_icdar2011_txt 'ground_truth_validation_icdar2011.txt'

echo 'Download test images...'
gdrive_download $data_test_tar_gz 'data_test.tar.gz'

echo 'Download test labels...'
gdrive_download_small $grount_truth_test_icdar2011_txt 'grount_truth_test_icdar2011.txt'

echo 'Extract training files...'
unrar x trainingsnippets_icdar.rar

echo 'Extract validation files'
tar -xvf validationsnippets_icdar.tar.gz

echo 'Extract test files'
tar -xvf data_test.tar.gz
