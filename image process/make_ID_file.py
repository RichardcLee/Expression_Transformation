import os
import argparse
import csv


parser = argparse.ArgumentParser()
parser.add_argument('-ip', '--image_path', type=str, help='path of dir with imgs files')
parser.add_argument('-op', '--output_path', type=str, help='path of ID file to output')
parser.add_argument('-m', '--mode', type=str, default="both", help='test or train or both')
args = parser.parse_args()

file_list = os.listdir(args.image_path)

if args.mode == "test" or args.mode == "both":
    with open(args.output_path + "/test_ids.csv", "w", newline="") as f:
        for file in file_list:
            csv_f = csv.writer(f)
            csv_f.writerow([file])

if args.mode == "train" or args.mode == "both":
    with open(args.output_path + "/train_ids.csv", "w", newline="") as f:
        for file in file_list:
            csv_f = csv.writer(f)
            csv_f.writerow([file])

print("Finish!")
