import net
import torch,os
from torchvision.transforms import transforms
import numpy as np
import matplotlib.pyplot as plt

def vis_cub():
	model=net.get_net(classes_num=200,channel_size=2048)
	model.load_state_dict(torch.load("./weights/cub/resnet50_cub_best_acc.pth"),strict=True)
	transform=transforms.Compose([transforms.Resize(size=(224,224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
	model.eval()
	img_name_list=os.listdir("./utils/ori/cub/")

	for img_name in img_name_list:
		old_img_path="./utils/ori/cub/"+img_name
		attention_img_path="./utils/new/cub/"+img_name
		model.get_attention_map(old_img_path,transform,attention_img_path)
		print("{} finished".format(old_img_path))

def vis_dogs():
	model=net.get_net(classes_num=120,channel_size=2048)
	model.load_state_dict(torch.load("./weights/dogs/resnet50_dogs_best_acc.pth"),strict=True)
	transform=transforms.Compose([transforms.Resize(size=(224,224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
	model.eval()
	img_name_list=os.listdir("./utils/ori/dogs/")

	for img_name in img_name_list:
		old_img_path="./utils/ori/dogs/"+img_name
		attention_img_path="./utils/new/dogs/"+img_name
		model.get_attention_map(old_img_path,transform,attention_img_path)
		print("{} finished".format(old_img_path))

def draw_hist(feature_vec):
	x=plt.hist(feature_vec,bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
		rwidth=0.8,facecolor="lightblue")
	for i in x[0]:
		print(i)
	plt.xlabel('Channel Feature after Global Average Pooling',fontsize=10)
	plt.ylabel('Number of Channels',fontsize=10)
	fig = plt.gcf()
	fig.set_size_inches(7.2, 4.2)
	plt.show()

def count_cub():
	model=net.get_net(classes_num=200,channel_size=2048)
	model.load_state_dict(torch.load("./weights/cub/resnet50_cub_best_acc.pth"),strict=True)
	transform=transforms.Compose([transforms.Resize(size=(224,224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
	model.eval()
	img_name_list=os.listdir("./utils/ori/cub/")
	feature_vec=[]
	for img_name in img_name_list:
		old_img_path="./utils/ori/cub/"+img_name
		vec=model.get_count_result(old_img_path,transform)
		feature_vec.append(vec)
	feature_vec=np.concatenate(feature_vec,axis=0)
	print(feature_vec.shape)
	draw_hist(feature_vec)

def get_cub_bounding_box():
	model=net.get_net(classes_num=200,channel_size=2048)
	model.load_state_dict(torch.load("./weights/cub/resnet50_cub_best_acc.pth"),strict=True)
	transform=transforms.Compose([transforms.Resize(size=(224,224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
	model.eval()
	img_name_list=os.listdir("./utils/ori/cub/")

	for img_name in img_name_list:
		old_img_path="./utils/ori/cub/"+img_name
		new_image_path="./utils/bnd/cub/"+img_name
		model.get_boundbox_image(old_img_path,transform,new_image_path)

def get_dogs_bounding_box():
	model=net.get_net(classes_num=120,channel_size=2048)
	model.load_state_dict(torch.load("./weights/dogs/resnet50_dogs_best_acc.pth"),strict=True)
	transform=transforms.Compose([transforms.Resize(size=(224,224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
	model.eval()
	img_name_list=os.listdir("./utils/ori/dogs/")

	for img_name in img_name_list:
		old_img_path="./utils/ori/dogs/"+img_name
		new_image_path="./utils/bnd/dogs/"+img_name
		model.get_boundbox_image(old_img_path,transform,new_image_path)

if __name__=="__main__":
	#get_cub_bounding_box()
	#get_dogs_bounding_box()
	#vis_dogs()
	#count_cub()
