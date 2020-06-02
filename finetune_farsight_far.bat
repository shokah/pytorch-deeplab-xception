C:\Users\gisstudent\.conda\envs\shai\python.exe train.py ^
 --backbone mobilenet ^
 --lr 0.0001 ^
 --num_class 100 ^
 --workers 4 ^
 --epochs 20 ^
 --batch-size 2 ^
 --min_depth 200 ^
 --max_depth 1000.1 ^
 --checkname my_deeplab_mobilenet_far_farsight ^
 --eval-interval 1 ^
 --dataset farsight ^
 --loss-type depth_loss ^
 --resume D:\Shai_Schneider\pytorch-deeplab-xception\models\deeplab-mobilenet.pth.tar ^
 --ft
 @pause