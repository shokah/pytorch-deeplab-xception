D:\Shai_Schneider\pytorch-deeplab-xception\venv\python.exe train.py ^
 --backbone mobilenet ^
 --lr 0.01 ^
 --workers 4 ^
 --epochs 40 ^
 --batch-size 4 ^
 --checkname deeplab-mobilenet_my ^
 --eval-interval 1 ^
 --dataset apollo ^
 --loss-type depth_loss ^
 --resume models/deeplab-mobilenet.pth.tar ^
 --ft
 @pause