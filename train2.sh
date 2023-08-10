#!/bin/bash
sbatch python3 train_semseg_ponderNethidden_only_trainitr.py --model PonderNethidden_sem_seg --test_area 5 --log_dir PonderNethidden_sem_seg_only_train_iterations_num_itr1  --num_itr 1 &
sbatch python3 train_semseg_ponderNethidden_only_trainitr.py --model PonderNethidden_sem_seg --test_area 5 --log_dir PonderNethidden_sem_seg_only_train_iterations_num_itr2  --num_itr 2 &
sbatch python3 train_semseg_ponderNethidden_only_trainitr.py --model PonderNethidden_sem_seg --test_area 5 --log_dir PonderNethidden_sem_seg_only_train_iterations_num_itr3  --num_itr 3 &
sbatch python3 train_semseg_ponderNethidden_only_trainitr.py --model PonderNethidden_sem_seg --test_area 5 --log_dir PonderNethidden_sem_seg_only_train_iterations_num_itr4  --num_itr 4 &
sbatch python3 train_semseg_ponderNethidden_only_trainitr.py --model PonderNethidden_sem_seg --test_area 5 --log_dir PonderNethidden_sem_seg_only_train_iterations_num_itr5  --num_itr 5 &
wait