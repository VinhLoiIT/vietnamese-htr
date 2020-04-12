#!/bin/sh
echo "Download dataset.."
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1MRcSXxMJoGKmKtmCgohKLWpu6q3rsXX5' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1MRcSXxMJoGKmKtmCgohKLWpu6q3rsXX5" -O rimes_line.tar.gz && rm -rf /tmp/cookies.txt

echo "Extract dataset"
tar -xvf rimes_line.tar.gz

echo "Create csv files"
cd RIMES_Line
python3 split.py

echo "Extract training images"
tar -xvf training_2011.tar

echo "Extract testing imagese"
tar -xvf eval_2011.tar

echo "Done"
