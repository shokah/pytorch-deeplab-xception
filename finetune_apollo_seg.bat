C:\Users\gisstudent\.conda\envs\shai\python.exe train.py ^
 --backbone mobilenet ^
 --lr 0.0001 ^
 --num_class 2 ^
 --workers 4 ^
 --epochs 20 ^
 --batch-size 2 ^
 --cut_point 100 ^
 --checkname my_deeplab_mobilenet_near ^
 --eval-interval 1 ^
 --dataset apollo_seg ^
 --loss-type ce ^
 --resume D:\Shai_Schneider\pytorch-deeplab-xception\models\deeplab-mobilenet.pth.tar ^
 --ft
 @pause