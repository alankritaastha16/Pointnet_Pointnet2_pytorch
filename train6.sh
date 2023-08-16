#!/bin/bash
#python3 train_partseg_iterativehidden_only_trainitr.py --model PonderNethidden_part_seg --log_dir iterativenewmodelshidden_part_seg_ssg_numitr1  --epoch 100 --num_itr 1 &
python3 train_partseg_iterativehidden_only_trainitr.py --model PonderNethidden_part_seg_ssg --log_dir iterativenewmodelshidden_part_seg_ssg_numitr2  --epoch 100 --num_itr 2 &
python3 train_partseg_iterativehidden_only_trainitr.py --model PonderNethidden_part_seg_ssg --log_dir iterativenewmodelshidden_part_seg_ssg_numitr3  --epoch 100 --num_itr 3 &
wait