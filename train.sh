#!/bin/bash
python3 train_partseg_ponderNethidden_only_trainitr.py --model pointnetPondernethidden_part_seg --log_dir pointnetpondernethidden_part_seg_ssg_numitr1  --epoch 100 --num_itr 1  &
python3 train_partseg_ponderNethidden_only_trainitr.py --model pointnetPondernethidden_part_seg --log_dir pointnetpondernethidden_part_seg_ssg_numitr2  --epoch 100 --num_itr 2  &
python3 train_partseg_ponderNethidden_only_trainitr.py --model pointnetPondernethidden_part_seg --log_dir pointnetpondernethidden_part_seg_ssg_numitr3  --epoch 100 --num_itr 3  &
python3 train_partseg_ponderNethidden_only_trainitr.py --model pointnetPondernethidden_part_seg --log_dir pointnetpondernethidden_part_seg_ssg_numitr4  --epoch 100 --num_itr 4  &
python3 train_partseg_ponderNethidden_only_trainitr.py --model pointnetPondernethidden_part_seg --log_dir pointnetpondernethidden_part_seg_ssg_numitr5  --epoch 100 --num_itr 5  &
wait