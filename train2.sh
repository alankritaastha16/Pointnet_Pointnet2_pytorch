#!/bin/bash
python3 train_partseg_iterativemodels.py --model pointnetPondernet_part_seg  --log_dir pointnetPondernetiterative_part_seg_ssg_numitr1  --epoch 100 --num_itr 1 &
python3 train_partseg_iterativemodels.py --model pointnetPondernet_part_seg  --log_dir pointnetPondernetiterative_part_seg_ssg_numitr2  --epoch 100 --num_itr 2 &
python3 train_partseg_iterativemodels.py --model pointnetPondernet_part_seg  --log_dir pointnetPondernetiterative_part_seg_ssg_numitr3  --epoch 100 --num_itr 3 &
wait
