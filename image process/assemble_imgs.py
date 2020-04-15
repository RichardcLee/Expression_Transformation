# 写在前面： 这是个危险操作，请认真检查路径
import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-ip', '--image_path', default="xxx", type=str, help='path of dir with imgs files')
parser.add_argument('-op', '--output_path', default="xxx", type=str, help='path of dir to store moved images')
args = parser.parse_args()

input_dir = args.image_path   # 源路径
output_dir = args.output_path  # 存储路径
file_type = (".jpg", ".jpeg", ".png", ".bmp")  # 需要移动的图片格式

key = input("Make sure you wanna move images？Y/N\n")
if key != "Y" and key != "y":
    print("exit.")
    exit(-1)
print("input_dir: '%s', output_dir: '%s'" % (input_dir, output_dir))
key = input("All paths are right？Y/N\n")
if key != "Y" and key != "y":
    print("exit.")
    exit(-1)


# 遍历input_dir下的所有文件，将符合file_type中指定的类型的文件统一移动到output_dir下
for folderName, subfolders, filenames in os.walk(input_dir):
    print(folderName)
    for filename in filenames:
        if filename.endswith(file_type):
            print(filename)
            try:
                shutil.move(folderName + '\\' + filename, output_dir + '\\' + filename)
            except OSError as e:
                print(e)
