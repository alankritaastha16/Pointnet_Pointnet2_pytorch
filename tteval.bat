@echo off
start /B python test_partseg_iterativemodels.py --log_dir pointnetPondernetiterative_part_seg_ssg_numitr1 --datadir "C:\Users\alank\OneDrive - TUM\Master Thesis\Algorithms\data" --num_itr 1
start /B python test_partseg_iterativemodels.py --log_dir pointnetPondernetiterative_part_seg_ssg_numitr2 --datadir "C:\Users\alank\OneDrive - TUM\Master Thesis\Algorithms\data" --num_itr 2
start /B python test_partseg_iterativemodels.py --log_dir pointnetPondernetiterative_part_seg_ssg_numitr3 --datadir "C:\Users\alank\OneDrive - TUM\Master Thesis\Algorithms\data" --num_itr 3
