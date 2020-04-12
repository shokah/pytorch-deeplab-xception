C:\Users\gisstudent\.conda\envs\shai\python.exe train.py ^
 --backbone mobilenet ^
 --lr 0.001 ^
 --num_class 1 ^
 --workers 4 ^
 --epochs 40 ^
 --batch-size 2 ^
 --min_depth 3 ^
 --max_depth 656 ^
 --checkname sigmoid_grad_loss_full_train ^
 --eval-interval 1 ^
 --dataset apollo ^
 --loss-type depth_sigmoid_grad_loss
 @pause