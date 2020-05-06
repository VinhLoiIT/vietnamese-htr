#!/bin/bash

mkdir -p VNOnDB
cd VNOnDB

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1TvCk9451FyQ6ru77oXNrwpYCp5phIDh0' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1TvCk9451FyQ6ru77oXNrwpYCp5phIDh0" -O line.tar.gz && rm -rf /tmp/cookies.txt

tar -xvf line.tar.gz
cd line
mv ../line.tar.gz .
tar -xvf train_line.tar.gz
tar -xvf test_line.tar.gz
tar -xvf validation_line.tar.gz
