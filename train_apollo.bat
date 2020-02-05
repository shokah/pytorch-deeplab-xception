C:\Users\gisstudent\.conda\envs\shai\python.exe train.py ^
 --backbone mobilenet ^
 --lr 0.01 ^
 --num_class 100 ^
 --split_method ud ^
 --workers 4 ^
 --epochs 40 ^
 --batch-size 2 ^
 --min_depth 0 ^
 --max_depth 50 ^
 --checkname debug_run ^
 --eval-interval 1 ^
 --dataset apollo ^
 --loss-type depth_loss
 @pause