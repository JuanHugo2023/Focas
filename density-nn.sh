#!/bin/bash

# export LD_LIBRARY_PATH="$HOME/src/cuda"
echo $LD_LIBRARY_PATH
python density-nn.py --model count.cfg --load yolo.weights --train --gpu 0.9 --batch 32 --ntrain 13 --nload 40
