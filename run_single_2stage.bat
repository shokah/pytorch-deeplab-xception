C:\Users\gisstudent\.conda\envs\shai\python.exe demo_two_stage.py ^
 --in-path D:\Shai_Schneider\Apollo_dataset\test\RGB\13-00\CLEAR_SKY\NO_DEGRADATION\With_Pedestrian\With_TrafficBarrier\Downtown\Traffic_117\0000000.jpg ^
 --out-path D:\Shai_Schneider\pytorch-deeplab-xception\examples\example_2stage.png ^
 --backbone mobilenet ^
 --num_class_near 100 ^
 --num_class_far 50 ^
 --min_depth_near 0 ^
 --max_depth_near 100 ^
 --min_depth_far 100 ^
 --max_depth_far 656 ^
 --ckpt_near D:\Shai_Schneider\pytorch-deeplab-xception\run\apollo\my_deeplab_mobilenet_near\model_best.pth.tar ^
 --ckpt_far D:\Shai_Schneider\pytorch-deeplab-xception\run\apollo\my_deeplab_mobilenet_far\model_best.pth.tar ^
 --ckpt_seg D:\Shai_Schneider\pytorch-deeplab-xception\run\apollo_seg\my_deeplab_mobilenet_seg2class\model_best.pth.tar 
 @pause