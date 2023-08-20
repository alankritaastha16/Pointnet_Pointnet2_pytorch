#!/bin/bash
python3 test_semseg_PonderNet_Choppedscenes.py  --log_dir pointnetPonderNet_sem_seg_num_itr1  --num_itr 1 &
python3 test_semseg_PonderNet_Choppedscenes.py  --log_dir pointnetPonderNet_sem_seg_num_itr2  --num_itr 2 &
python3 test_semseg_PonderNet_Choppedscenes.py  --log_dir pointnetPonderNet_sem_seg_num_itr3  --num_itr 3 &
python3 test_semseg_PonderNet_Choppedscenes.py  --log_dir pointnetPonderNet_sem_seg_num_itr4  --num_itr 4 &
python3 test_semseg_PonderNet_Choppedscenes.py  --log_dir pointnetPonderNet_sem_seg_num_itr5  --num_itr 5 &
python3 test_semseg_PonderNethidden_Choppedscenes.py  --log_dir PonderNethidden_sem_seg_only_train_iterations_num_itr1  --num_itr 1 &
python3 test_semseg_PonderNethidden_Choppedscenes.py  --log_dir PonderNethidden_sem_seg_only_train_iterations_num_itr2  --num_itr 2 &
python3 test_semseg_PonderNethidden_Choppedscenes.py  --log_dir PonderNethidden_sem_seg_only_train_iterations_num_itr3  --num_itr 3 &
python3 test_semseg_PonderNethidden_Choppedscenes.py  --log_dir PonderNethidden_sem_seg_only_train_iterations_num_itr4  --num_itr 4 &
python3 test_semseg_PonderNethidden_Choppedscenes.py  --log_dir PonderNethidden_sem_seg_only_train_iterations_num_itr5  --num_itr 5 &
python3 test_semseg_PonderNethidden_Choppedscenes.py  --log_dir pointnetPonderNethidden_sem_seg_num_itr1  --num_itr 1 &
python3 test_semseg_PonderNethidden_Choppedscenes.py  --log_dir pointnetPonderNethidden_sem_seg_num_itr2  --num_itr 2 &
python3 test_semseg_PonderNethidden_Choppedscenes.py  --log_dir pointnetPonderNethidden_sem_seg_num_itr3  --num_itr 3 &
python3 test_semseg_PonderNethidden_Choppedscenes.py  --log_dir pointnetPonderNethidden_sem_seg_num_itr4  --num_itr 4 &
python3 test_semseg_PonderNethidden_Choppedscenes.py  --log_dir pointnetPonderNethidden_sem_seg_num_itr5  --num_itr 5 &
wait
