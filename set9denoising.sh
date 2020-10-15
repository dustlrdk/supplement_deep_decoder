#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=$1
python set9_exp.py 'image_F16_512rgb'
python set9_exp.py 'image_Lena512rgb'
python set9_exp.py 'kodim12'
python set9_exp.py 'kodim03'
python set9_exp.py 'image_House256rgb'
python set9_exp.py 'kodim02'
python set9_exp.py 'kodim01'
python set9_exp.py 'image_Peppers512rgb'
python set9_expdenoising.py 'image_Baboon512rgb'