import os
import shutil

split_file = r"D:\Shai_Schneider\farsight_513x513\test.txt"
rgb_folder = r"D:\Shai_Schneider\farsight_513x513\img"
depth_folder = r"D:\Shai_Schneider\farsight_513x513\depth-in-meters"
rgb_target_folder = r"D:\Shai_Schneider\farsight_513x513\test\RGB"
depth_target_folder = r"D:\Shai_Schneider\farsight_513x513\test\Depth"

f = open(split_file, "r")
for line in f:
    line=line.strip()
    rgb_cur_path = os.path.join(rgb_folder, line)
    depth_cur_path = os.path.join(depth_folder, line)

    rgb_new_path = os.path.join(rgb_target_folder, line)
    depth_new_path = os.path.join(depth_target_folder, line)

    shutil.copy(rgb_cur_path, rgb_new_path)
    shutil.copy(depth_cur_path, depth_new_path)

    print(rgb_cur_path)
    print(rgb_new_path)
    print(80*'*')
    print(depth_cur_path)
    print(depth_new_path)
    print(80 * '*')
    print(80 * '*')
