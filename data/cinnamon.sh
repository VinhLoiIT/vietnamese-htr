#!/bin/bash
echo Download dataset

train_tar_gz='1xRqou11EIeQqyFwAXrb3UKdYwI6nUJhM'
test_tar_gz='1cvIMpi-zntZI1CFJlH-015BwwQN_d3nG'
csv_tar_gz='1WKiSEdyXyOQuoSebjCjf0pn69U1og5hA'

function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

function gdrive_download_small () {
  wget --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O $2
}


mkdir -p Cinnamon
cd Cinnamon

gdrive_download $train_tar_gz train.tar.gz
gdrive_download $test_tar_gz test.tar.gz
gdrive_download_small $csv_tar_gz csv.tar.gz

echo 'Extract train data'
tar -xvf train.tar.gz
tar -xvf test.tar.gz
tar -xvf csv.tar.gz

echo 'Done'