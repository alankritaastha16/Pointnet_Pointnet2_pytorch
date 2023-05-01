#!/bin/bash

#sbatch python3 train_partseg.py --model pointnet2_part_seg_ssg --log_dir pointnet2_part_seg_ssg
#sbatch python3 train_partseg.py --model pointnet2_part_seg_msg --log_dir pointnet2_part_seg_msg
#sbatch python3 train_semseg.py --model pointnet2_sem_seg --test_area 5 --log_dir pointnet2_sem_seg_subet8000 --subset 8000
#sbatch python3 train_semseg.py --model pointnet2_sem_seg_msg --test_area 5 --log_dir pointnet2_sem_seg_msg
#sbatch python3 train_partseg_ponderNet.py --model PonderNet_part_seg_ssg --log_dir PonderNet_part_seg_ssg_only_train_iterations --epoch 50
#sbatch python3 train_semseg_ponderNet.py --model PonderNet_sem_seg --test_area 5 --log_dir PonderNet_sem_seg_subset8000_only_train_iterarions --subset 8000
#sbatch python3 train_partseg_ponderNet.py --model PonderNet_part_seg_msg --log_dir PonderNet_part_seg_msg
#sbatch python3 train_semseg_ponderNet.py --model PonderNet_sem_seg_msg --test_area 5 --log_dir PonderNet_sem_seg_msg
#srun python3 test_semseg_Choppedscenes.py --log_dir pointnet2_sem_seg_only_trainitr
#srun python3 test_partseg.py --log_dir pointnet2_part_seg_msg_only_trainitr_75epochs
#srun python3 test_partseg_PonderNet.py --log_dir PonderNet_part_seg_ssg_onlytrain_iterations_num_itr5
sbatch python3 train_partseg_ponderNethidden_only_trainitr.py --model PonderNethidden_part_seg_ssg --log_dir PonderNethidden_part_seg_ssg_only_train_iterations --epoch 100 --num_itr 1