#!/bin/bash
python3 train_semseg_ponderNethidden_only_trainitr.py --model pointnetPondernethidden_sem_seg --test_area 5 --log_dir pointnetPonderNethidden_sem_seg_num_itr1  --num_itr 1 &
python3 train_semseg_ponderNethidden_only_trainitr.py --model pointnetPondernethidden_sem_seg --test_area 5 --log_dir pointnetPonderNethidden_sem_seg_num_itr2  --num_itr 2 &
python3 train_semseg_ponderNethidden_only_trainitr.py --model pointnetPondernethidden_sem_seg --test_area 5 --log_dir pointnetPonderNethidden_sem_seg_num_itr3  --num_itr 3 &
python3 train_semseg_ponderNethidden_only_trainitr.py --model pointnetPondernethidden_sem_seg --test_area 5 --log_dir pointnetPonderNethidden_sem_seg_num_itr4  --num_itr 4 &
python3 train_semseg_ponderNethidden_only_trainitr.py --model pointnetPondernethidden_sem_seg --test_area 5 --log_dir pointnetPonderNethidden_sem_seg_num_itr5  --num_itr 5 &
wait
