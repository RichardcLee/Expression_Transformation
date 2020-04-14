# 写在前面： 这是个危险操作，请认真检查路径
import os
import shutil


input_dir = r"D:\我的文件\数据集\cohn-kanade/"   # 源路径
output_dir = r"C:\Users\81955\Desktop\cohn-kanade/"  # 存储路径
file_type = (".jpg", ".jpeg", ".png", ".bmp")  # 需要移动的图片格式

key = input("是否要批量移动图片？Y/N\n")
if key != "Y" and key != "y":
    print("已退出")
    exit(-1)
print("input_dir: '%s', output_dir: '%s'" % (input_dir, output_dir))
key = input("请检查路径是否正确？Y/N\n")
if key != "Y" and key != "y":
    print("已退出")
    exit(-1)


# 遍历input_dir下的所有文件，将符合file_type中指定的类型的文件统一移动到output_dir下
for folderName, subfolders, filenames in os.walk(input_dir):
    print(folderName)
    for filename in filenames:
        if filename.endswith(file_type):
            print(filename)
            try:
                shutil.move(folderName + '\\' + filename, output_dir + '\\'+ filename)
            except OSError as e:
                print(e)
