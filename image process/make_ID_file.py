import os
import argparse
import csv


parser = argparse.ArgumentParser()
parser.add_argument('-ip', '--image_path', type=str, help='path of dir with imgs files')
parser.add_argument('-op', '--output_path', type=str, help='path of ID file to output')
parser.add_argument('-m', '--mode', type=str, default="both", help='test or train or both')
parser.add_argument('-lim1', '--limits_train', type=int, default="146285", help='train set size. 80%')
parser.add_argument('-lim2', '--limits_test', type=int, default="36572", help='test set size. 20%')
args = parser.parse_args()

file_list = os.listdir(args.image_path)

# 自动生成ID文件
idx = 0
if args.mode == "train" or args.mode == "both":
    with open(args.output_path + "/train_ids.csv", "w", newline="") as f:
        for file in file_list:
            if idx == args.limits_train:
                break
            csv_f = csv.writer(f)
            csv_f.writerow([file])
            idx += 1

idx = 0
if args.mode == "test" or args.mode == "both":
    with open(args.output_path + "/test_ids.csv", "w", newline="") as f:
        for file in file_list:
            if idx < args.limits_train:
                idx += 1
                continue
            csv_f = csv.writer(f)
            csv_f.writerow([file])


print("Finish!")
