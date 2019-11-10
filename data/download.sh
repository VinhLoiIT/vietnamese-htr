wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1MYj7QkpbsvLIuowloG0vlAW5uvAYCr75' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1MYj7QkpbsvLIuowloG0vlAW5uvAYCr75" -O word_train.zip && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1sRT5sJZLfFxEJnq6LAgRSirUIqEdwogK' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1sRT5sJZLfFxEJnq6LAgRSirUIqEdwogK" -O word_test.zip && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1NcWPK40JCm2b0L460rRZY-9sbTctDJnh' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1NcWPK40JCm2b0L460rRZY-9sbTctDJnh" -O word_val.zip && rm -rf /tmp/cookies.txt

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1G-M6aw5Pbog0CBp-ewpgvFFlrBoR3JI7' -O test_word.csv

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=14AD4ToIKQAm6GGCxlQTsWvSV4XBLR8VS' -O train_word.csv

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1QkVoe591EAY71XBQz1jY-Y2ZV0y3vyog' -O validation_word.csv

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1qIJNpvl7lKSs7sQOwqLCLncWpZylwxsI' -O all_word.csv
