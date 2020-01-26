venv\Scripts\python.exe train.py ^
 --backbone mobilenet ^
 --lr 0.01 ^
 --workers 4 ^
 --epochs 40 ^
 --batch-size 2 ^
 --checkname deeplab-mobilenet_my ^
 --eval-interval 1 ^
 --dataset apollo ^
 --loss-type depth_loss ^
 --resume models/deeplab-mobilenet.pth.tar ^
 --no-cuda ^
 --ft
 @pause