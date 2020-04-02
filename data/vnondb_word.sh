train_word_tar_gz='1Rv-7lRdvWqKnRL-QTEUPhHMfIpMpaN1e'
test_word_tar_gz='1t74UpNBTo2cJ6up4G2kmPlPSALwvMtVO'
validation_word_tar_gz='1TUUNvVIaBU5lUPB4Fc6yJrk5_Jmyocg2'
csv_tar_gz='1hfo5-qS0AhsXEoDTQihIsUFa1wn5f34k'

function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

gdrive_download $train_word_tar_gz train_word.tar.gz
gdrive_download $test_word_tar_gz test_word.tar.gz
gdrive_download $validation_word_tar_gz validation_word.tar.gz
gdrive_download $csv_tar_gz csv.tar.gz