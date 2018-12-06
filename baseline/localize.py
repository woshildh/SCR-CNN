import net
import torch,os
from torchvision.transforms import transforms
import numpy as np
class Rect():
	'''
	定义矩形:first是左上角,second是右下角
	'''
	def __init__(self,x1,y1,x2,y2):
		self.first=(x1,y1)
		self.second=(x2,y2)

def calc_pos(val_path="../data/cub/CUB200/images/val/",log_path="./loc.txt"):
	model=net.get_net(classes_num=200,channel_size=2048)
	model.load_state_dict(torch.load("./weights/cub/resnet50_cub_best_acc.pth"),strict=True)
	transform=transforms.Compose([transforms.Resize(size=(224,224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
	model.eval()
	dir_list=os.listdir(val_path)
	log_list=[]
	for dir_ in dir_list:
		imgae_list=os.listdir(val_path+"/"+dir_)
		for image in imgae_list:
			image_path=val_path+"/"+dir_+"/"+image
			p1,p2=model.get_boundbox_image(image_path,transform)
			log_list.append([image,p1,p2])
			print("{} finished".format(image_path))
	with open(log_path,"w",encoding="utf-8") as file:
		for item in log_list:
			image,p1,p2=item
			content="{},{},{},{},{}".format(image,p1[0],p1[1],p2[0],p2[1])+"\n"
			file.write(content)

def calc_iou(r1,r2):
	'''
	求矩形r1和r2的交并比
	'''
	#先求出两个矩形各自的宽和高
	width1=r1.second[0]-r1.first[0]
	height1=r1.second[1]-r1.first[1]

	width2=r2.second[0]-r2.first[0]
	height2=r2.second[1]-r2.first[1]

	#求出最大的宽和最大的高
	xmax=max(r1.second[0],r2.second[0])
	xmin=min(r1.first[0],r2.first[0])
	ymax=max(r1.second[1],r2.second[1])
	ymin=min(r1.first[1],r2.first[1])

	#求最大宽和最大高
	width=xmax-xmin
	height=ymax-ymin

	#求交集部分的宽和高
	inter_width=width1+width2-width
	inter_height=height1+height2-height

	if inter_height<=0 or inter_width<=0:
		return 0
	else:
		area=inter_height*inter_width
		area1=width1*height1
		area2=width2*height2
		iou_num=area/(area1+area2-area)
		return iou_num

def get_result(log_path="./loc.txt",scale_txt_path="./scale.txt",
		dataset_path="../data/cub/CUB200/",iou_gap=0.5):
	
	images_txt=dataset_path+"images.txt"
	bnd_txt=dataset_path+"bounding_boxes.txt"

	#build image_name and image_id map
	name_id_map=dict()
	lines=open(images_txt,"r").readlines()
	for line in lines:
		line=line.strip()
		if line=="":
			break
		else:
			id_,name=line.split(" ")
			name=name.split("/")[-1]
			name_id_map[name]=id_
	#build image_id and scales map
	image_scale_map=dict()
	lines=open(scale_txt_path,"r").readlines()
	for line in lines:
		line=line.strip()
		if line is not "":
			name,h,w=line.split(",")
			image_scale_map[name_id_map[name]]=(float(h),float(w))
	#build image_id and boundingbox map
	id_bnd_map=dict()
	lines=open(bnd_txt,"r").readlines()
	for line in lines:
		line=line.strip()
		if line=="":
			continue
		else:
			id_,x1,y1,x2,y2=line.split(" ")
			x1,y1,x2,y2=float(x1),float(y1),float(x2),float(y2)
			try:
				height_scaler=image_scale_map[id_][0]/224.0
				width_scaler=image_scale_map[id_][1]/224.0
			except:
				height_scaler,width_scaler=1,1
			x1,x2=x1/width_scaler,x2/width_scaler
			y1,y2=y1/height_scaler,y2/height_scaler
			id_bnd_map[id_]=(x1,y1,x2,y2)
	#read log file
	lines=open(log_path,"r").readlines()
	count=0.001
	correct=0
	for line in lines:
		line=line.strip()
		if line=="":
			continue
		else:
			name,x1,y1,x2,y2=line.split(",")
			x1,y1,x2,y2=float(x1),float(y1),float(x2),float(y2)
			r1=Rect(x1,y1,x2,y2)
			x1,y1,x2,y2=id_bnd_map[name_id_map[name]]
			r2=Rect(x1,y1,x2,y2)
			iou=calc_iou(r1,r2)
			if iou>0.3:
				correct+=1
			count+=1
	print("The loc acc is:{:.4f}".format(correct/count))

if __name__=="__main__":
	#calc_pos()
	get_result(log_path="./loc.txt",iou_gap=0.3)
