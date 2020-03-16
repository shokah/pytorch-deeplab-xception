C:\Users\gisstudent\.conda\envs\shai\python.exe train.py ^
 --backbone mobilenet ^
 --lr 0.0001 ^
 --num_class 1 ^
 --workers 4 ^
 --epochs 20 ^
 --batch-size 2 ^
 --min_depth 3 ^
 --max_depth 656 ^
 --checkname my_deeplab_mobilenet_full_sigmoid ^
 --eval-interval 1 ^
 --dataset apollo ^
 --loss-type depth_sigmoid_loss ^
 --resume D:\Shai_Schneider\pytorch-deeplab-xception\models\deeplab-mobilenet.pth.tar ^
 --ft
 @pause