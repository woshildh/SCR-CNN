import cv2,os
img_list=os.listdir("./dogs/")
for img in img_list:
	image=cv2.imread("./dogs/"+img)
	res=cv2.resize(image,(224,224),interpolation=cv2.INTER_NEAREST)
	cv2.imwrite("./dogs/"+img,res)