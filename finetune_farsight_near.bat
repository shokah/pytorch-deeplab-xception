C:\Users\gisstudent\.conda\envs\shai\python.exe train.py ^
 --backbone mobilenet ^
 --lr 0.0001 ^
 --num_class 150 ^
 --workers 4 ^
 --epochs 20 ^
 --batch-size 2 ^
 --min_depth 1 ^
 --max_depth 200 ^
 --checkname my_deeplab_mobilenet_near_farsight ^
 --eval-interval 1 ^
 --dataset farsight ^
 --loss-type depth_loss ^
 --resume D:\Shai_Schneider\pytorch-deeplab-xception\models\deeplab-mobilenet.pth.tar ^
 --ft
 @pause