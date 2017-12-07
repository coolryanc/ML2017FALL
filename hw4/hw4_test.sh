#!/bin/bash
wget 'https://www.dropbox.com/s/zqnl15otrt9i1co/currentbest1.h5?dl=1'
mv currentbest1.h5?dl=1 currentbest1.h5
wget 'https://www.dropbox.com/s/2ams7m78ttkmcog/currentbest.h5?dl=1'
mv currentbest.h5?dl=1 currentbest.h5
wget 'https://www.dropbox.com/s/841iotdgy9dj66f/words.txt?dl=1'
mv words.txt?dl=1 words.txt
python3 predict.py $1 $2
