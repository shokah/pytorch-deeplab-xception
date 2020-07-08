C:\Users\gisstudent\.conda\envs\shai\python.exe train_with_depth.py ^
 --backbone mobilenet ^
 --lr 0.0001 ^
 --num_class 1 ^
 --workers 4 ^
 --epochs 20 ^
 --batch-size 2 ^
 --min_depth 1 ^
 --max_depth 1000.1 ^
 --checkname my_deeplab_mobilenet_full_mixed_with_aprox_depth ^
 --eval-interval 1 ^
 --dataset farsight ^
 --loss-type depth_sigmoid_loss ^
 --resume D:\Shai_Schneider\pytorch-deeplab-xception\models\deeplab-mobilenet.pth.tar ^
 --ckpt D:\Shai_Schneider\pytorch-deeplab-xception\run\apollo\my_deeplab_mobilenet_full_apollo_01\model_best.pth.tar ^
 --ft
 @pause