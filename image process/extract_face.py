from PIL import Image
import os
import json
import numpy as np


class FaceExtracter(object):
    def __init__(self, image_path: str, output_path: str, boundbox_store_path: str, final_size: [tuple, list]):
        # path of image want to extract，note: all images should be placed in sub-dir imgs
        self.image_path = image_path + "/"
        self.output_path = output_path + "/"
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.final_size = final_size
        self.face_bound_dict = {}  # 存储人脸边界
        with open(boundbox_store_path, "r") as f:
            self.face_bound_dict = json.loads(f.read())
        if self.face_bound_dict == {}:
            raise Exception("can't load face bound！")

    # def extractAllFace(self):  # 提取图片集中的所有脸
    #     image_name_list = os.listdir(self.image_path)
    #     for image_name in image_name_list:
    #         # 加载图片 (.jpg, .png, etc) 到numpy矩阵
    #         image = face_recognition.load_image_file(self.image_path + image_name)
    #         # 返回包含图像中人脸的边界的列表 eg.[(68, 211, 175, 103), ...]
    #         face_locations = face_recognition.face_locations(image)  # use GPU: model="cnn"
    #
    #         print("found {} face(s) in this {}.".format(len(face_locations), image_name))
    #
    #         for idx, face_location in enumerate(face_locations):
    #             top, right, bottom, left = face_location
    #             face_image = image[top:bottom, left:right]
    #             pil_image = Image.fromarray(face_image)
    #             pil_image = pil_image.resize(self.final_size, Image.ANTIALIAS)
    #
    #             if not os.path.exists(self.output_path):
    #                 os.makedirs(self.output_path)
    #
    #             pil_image.save(self.output_path + image_name.replace(".", "_%d." % idx))

    def extractOneFace(self):
        image_name_list = os.listdir(self.image_path)
        face_cnt = 0
        message = "| "

        for image_name in image_name_list:
            try:
                face_bound = self.face_bound_dict[image_name][0]
                top, right, bottom, left = face_bound
                image = Image.open(self.image_path + image_name)
                image_array = np.array(image)
                face_array = image_array[top:bottom, left:right]
                face_image = Image.fromarray(face_array)
                face_image = face_image.resize(self.final_size, Image.ANTIALIAS)
                face_image.save(self.output_path + image_name)

                print("Extract and save face: [%s]" % image_name)
                face_cnt += 1

            except Exception as e:  # 未检测到人脸，注意这不代表该图中真的没有人脸
                print(str(e))
                message += image_name + " | "
                print("No face in %s！" % image_name)

        print("Process finish！Total Image: [%d], Total Face: [%d], No face detect in %s" % (len(image_name_list), face_cnt, message))


# 因为本地无法安装face_recognition，只能在colab运行
if __name__ == '__main__':
    extracter = FaceExtracter(r'C:\Users\81955\Desktop\Expression_Transformation\datasets\face\Original_Img',
                              r"C:\Users\81955\Desktop\Expression_Transformation\datasets\face\imgs",
                              r"C:\Users\81955\Desktop\Expression_Transformation\datasets\face/face_bound.json", (200, 200))
    # extracter.extractAllFace()
    extracter.extractOneFace()
