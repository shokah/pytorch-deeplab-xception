C:\Users\gisstudent\.conda\envs\shai\python.exe test.py ^
 --backbone mobilenet ^
 --num_class 150 ^
 --num_class2 100 ^
 --min_depth 1 ^
 --max_depth 1000.1 ^
 --cut_point 200 ^
 --loss-type depth_multi_dnn ^
 --batch-size 2 ^
 --dataset farsight ^
 --ckpt D:\Shai_Schneider\pytorch-deeplab-xception\run\farsight\my_deeplab_mobilenet_near_farsight\model_best.pth.tar ^
 --ckpt2 D:\Shai_Schneider\pytorch-deeplab-xception\run\farsight\my_deeplab_mobilenet_far_farsight\model_best.pth.tar ^
 --ckpt_seg D:\Shai_Schneider\pytorch-deeplab-xception\run\farsight_seg\my_deeplab_mobilenet_seg2class_farsight\model_best.pth.tar 
 
 @pause