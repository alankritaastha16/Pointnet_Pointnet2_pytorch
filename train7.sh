#!/bin/bash
python3 train_partseg_iterativehidden_only_trainitr.py --model pointnetPondernethidden_part_seg --log_dir pointnetpondernetiterativehidden_part_seg_ssg_numitr4 --epoch 100 --num_itr 4 &
python3 train_partseg_iterativehidden_only_trainitr.py --model pointnetPondernethidden_part_seg --log_dir pointnetpondernetiterativehidden_part_seg_ssg_numitr5 --epoch 100 --num_itr 5 &
python3 train_partseg_iterativehidden_only_trainitr.py --model pointnetPondernethidden_part_seg --log_dir pointnetpondernetiterativehidden_part_seg_ssg_numitr7 --epoch 100 --num_itr 7 &
python3 train_partseg_iterativehidden_only_trainitr.py --model pointnetPondernethidden_part_seg --log_dir pointnetpondernetiterativehidden_part_seg_ssg_numitr9 --epoch 100 --num_itr 9 &
python3 train_partseg_iterativehidden_only_trainitr.py --model pointnetPondernethidden_part_seg --log_dir pointnetpondernetiterativehidden_part_seg_ssg_numitr10 --epoch 100 --num_itr 10 &
python3 train_partseg_iterativehidden_only_trainitr.py --model pointnetPondernethidden_part_seg --log_dir pointnetpondernetiterativehidden_part_seg_ssg_numitr15 --epoch 100 --num_itr 15 &
python3 train_partseg_iterativehidden_only_trainitr.py --model pointnetPondernethidden_part_seg --log_dir pointnetpondernetiterativehidden_part_seg_ssg_numitr20 --epoch 100 --num_itr 20 &
wait