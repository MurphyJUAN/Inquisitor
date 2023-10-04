DATA_FILE_ID="1qa9lIbZX2YrngtOeNoqPy_qThH93-Pg3"
DATA_FILE_NAME="train_test_data_keywords.zip"

# request file from google drive
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=${FILE_ID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${DATA_FILE_ID}" -O ${DATA_FILE_NAME} && rm -rf /tmp/cookies.txt
