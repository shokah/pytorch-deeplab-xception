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
 --checkname deeplab-mobilenet_my ^
 --eval-interval 1 ^
 --dataset apollo ^
 --loss-type depth_loss ^
 --resume models/deeplab-mobilenet.pth.tar ^
 --ft
 @pause