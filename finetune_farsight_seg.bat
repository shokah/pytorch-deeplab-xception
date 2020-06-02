C:\Users\gisstudent\.conda\envs\shai\python.exe train.py ^
 --backbone mobilenet ^
 --lr 0.0001 ^
 --num_class 2 ^
 --workers 4 ^
 --epochs 20 ^
 --batch-size 2 ^
 --min_depth -1 ^
 --max_depth 9999 ^
 --cut_point 200 ^
 --checkname my_deeplab_mobilenet_seg2class_farsight ^
 --eval-interval 1 ^
 --dataset farsight_seg ^
 --loss-type ce ^
 --resume D:\Shai_Schneider\pytorch-deeplab-xception\models\deeplab-mobilenet.pth.tar ^
 --ft
 @pause