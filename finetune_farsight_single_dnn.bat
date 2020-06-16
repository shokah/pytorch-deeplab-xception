C:\Users\gisstudent\.conda\envs\shai\python.exe train.py ^
 --backbone mobilenet ^
 --lr 0.01 ^
 --num_class 150 ^
 --num_class2 100 ^
 --cut_point 200 ^
 --workers 4 ^
 --epochs 20 ^
 --batch-size 2 ^
 --min_depth 1 ^
 --max_depth 1000.1 ^
 --checkname my_deeplab_mobilenet_single_dnn_farsight ^
 --eval-interval 1 ^
 --dataset farsight ^
 --loss-type depth_loss_two_distributions ^
 --resume D:\Shai_Schneider\pytorch-deeplab-xception\models\deeplab-mobilenet.pth.tar ^
 --ft
 @pause