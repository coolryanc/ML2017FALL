#!/bin/bash
cd ./model
unzip Model_400_5.h5.zip
unzip Model_300_4.h5.zip
unzip Model_250_4.h5.zip
cd ..
python3 predict.py $1 $2
