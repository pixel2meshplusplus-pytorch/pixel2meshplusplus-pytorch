#!/usr/bin/env bash

FILE_ID='1JsLda6ouINwO1PXF2H-UKAriTk7E7F4M' 
OUTPUT_NAME='p2mppdata.7z.001'
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=${FILE_ID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILE_ID}" -O $OUTPUT_NAME && rm -rf /tmp/cookies.txt

FILE_ID='1OlRhC5ZTDJ_h9IX_U7aiSyn8aFvdsMVh' 
OUTPUT_NAME='p2mppdata.7z.002'
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=${FILE_ID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILE_ID}" -O $OUTPUT_NAME && rm -rf /tmp/cookies.txt

FILE_ID='1bbqSjYGfVo_fZEWJYcrE6qQWX5JSgEMl' 
OUTPUT_NAME='p2mppdata.7z.003'
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=${FILE_ID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILE_ID}" -O $OUTPUT_NAME && rm -rf /tmp/cookies.txt