from PIL import Image
import os
import numpy as np
import json


class FaceSplitJoin(object):
    def __init__(self, bg_im_ph: str, f_im_ph: str, fb_ph: str, op_ph: str):
        self.background_img_path = bg_im_ph + "/"  # 背景板图片路径
        self.face_img_path = f_im_ph + "/"  # 生成假脸的路径,两者文件名（id）需一一对应！！！

        self.output_path = op_ph + "/"  # 拼接后的图片输出路径
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.face_bound_dict = {}  # 存储人脸边界
        with open(fb_ph, "r") as f:
            self.face_bound_dict = json.loads(f.read())
        if self.face_bound_dict == {}:
            raise Exception("can't load face bound！")

    def split_join(self):
        image_name_list = os.listdir(self.background_img_path)

        for image_name in image_name_list:
            background_image = Image.open(self.background_img_path + image_name)  # 打开背景板
            try:
                face_bound = self.face_bound_dict[image_name][0]
                top, right, bottom, left = face_bound

                fake_face_image = Image.open(self.face_img_path + image_name)
                # 必须调整尺寸，因为神经网络模型输出统一尺寸
                fake_face_image = fake_face_image.resize((right - left, bottom - top), Image.ANTIALIAS)

                # 拼接
                background_image_array = np.array(background_image)
                background_image_array[top:bottom, left:right] = np.array(fake_face_image)

                image = Image.fromarray(background_image_array.astype('uint8')).convert('RGB')
                image.save("%s/sj_%s" % (self.output_path, image_name))
                print("Split join [%s] successfully!" % image_name)
            except Exception as e:
                # print("Error:", e)
                print("No key: [%s]" % image_name)

        print("[Finish]")


if __name__ == '__main__':
    SJ = FaceSplitJoin(r"C:\Users\81955\Desktop\Expression_Transformation\datasets\face\Original_Img", r"C:\Users\81955\Desktop\Expression_Transformation\results\Fake face\alpha_4",
                       r"C:\Users\81955\Desktop\Expression_Transformation\datasets\face/face_bound.json",
                       r"C:\Users\81955\Desktop\Expression_Transformation\datasets\face/Final face/")
    SJ.split_join()
