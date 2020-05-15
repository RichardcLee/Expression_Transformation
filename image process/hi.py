import cv2

# 读取本相对路径下的initial.bmp文件
image = cv2.imread ("1.jpg")
image.resize()
# 将image对应图像在图像窗口显示出来
cv2.imshow('initial',image)
# waitKey使窗口保持静态直到用户按下一个键

# 对图像进行阈值分割，阈值设定为80，得到二值化灰度图
ret,image1 = cv2.threshold(image,80,255,cv2.THRESH_BINARY)
cv2.imshow('grayscale',image1)

image2 = image1.copy()		# 复制图片
for i in range(0,image1.shape[0]):	#image.shape表示图像的尺寸和通道信息(高,宽,通道)
    for j in range(0,image1.shape[1]):
	image2[i,j]= 255 - image1[i,j]
cv2.imshow('colorReverse',image2)

# 边缘提取
img = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
canny_img_one = cv2.Canny(img,300,150)
canny_img_two = canny_img_one.copy()	# 复制图片
for i in range(0,canny_img_one.shape[0]):	#image.shape表示图像的尺寸和通道信息(高,宽,通道)
	for j in range(0,canny_img_one.shape[1]):
		canny_img_two[i,j]= 255 - canny_img_one[i,j]
cv2.imshow('edge',canny_img_two)
cv2.waitKey(0)