import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-ip', '--image_path', type=str, help='path of dir with imgs files')
parser.add_argument('-si', '--start_idx', type=int, default=0, help='start index of image name')
args = parser.parse_args()

file_list = os.listdir(args.image_path)
os.chdir(args.image_path)   # 切换当前工作目录

for idx, file in enumerate(file_list):  # 批量重命名
    os.rename(file, str(args.start_idx + idx) + "." + file.split(".")[-1])

print("Finish!")
