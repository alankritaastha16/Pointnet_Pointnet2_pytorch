#!/bin/bash
sbatch python3 train_partseg_ponderNethidden_only_trainitr.py --model pointNetPonderNethidden_part_seg --log_dir pointnetpondernethidden_part_seg_ssg_numitr1  --epoch 100 --num_itr 1  &
sbatch python3 train_partseg_ponderNethidden_only_trainitr.py --model pointNetPonderNethidden_part_seg --log_dir pointnetpondernethidden_part_seg_ssg_numitr2  --epoch 100 --num_itr 2  &
sbatch python3 train_partseg_ponderNethidden_only_trainitr.py --model pointNetPonderNethidden_part_seg --log_dir pointnetpondernethidden_part_seg_ssg_numitr3  --epoch 100 --num_itr 3  &
sbatch python3 train_partseg_ponderNethidden_only_trainitr.py --model pointNetPonderNethidden_part_seg --log_dir pointnetpondernethidden_part_seg_ssg_numitr4  --epoch 100 --num_itr 4  &
sbatch python3 train_partseg_ponderNethidden_only_trainitr.py --model pointNetPonderNethidden_part_seg --log_dir pointnetpondernethidden_part_seg_ssg_numitr5  --epoch 100 --num_itr 5  &
wait