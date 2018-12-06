import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from model import resnet_cifar,resnext_cifar,densenet_cifar,preact_resnet_cifar
from model import resnet_image,resnext_image,densenet_image
from PIL import Image
from torchvision.transforms import transforms
import ImageReader,math
import config
import numpy as np,cv2,time,re

def get_state_dict(cnn_weights_path):
	if "densenet" in cnn_weights_path:
		pattern = re.compile(
		r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
		state_dict = torch.load(cnn_weights_path)
		for key in list(state_dict.keys()):
			res = pattern.match(key)
			if res:
				new_key = res.group(1) + res.group(2)
				state_dict[new_key] = state_dict[key]
				del state_dict[key]
	else:
		state_dict=torch.load(cnn_weights_path)

	return state_dict


def get_features(cnn_weights_path=None):
	features=resnet_image.resnet50()
	if cnn_weights_path is not None:
		state_dict=get_state_dict(cnn_weights_path)
		features.load_state_dict(state_dict,
			strict=False)
		print("{} load succeed...".format(cnn_weights_path))
	return features

class Net(nn.Module):
	def __init__(self,classes_num=8,channel_size=1664,drop_rate=0
		,cnn_weights_path=None):
		super(Net,self).__init__()
		self.classes_num=classes_num
		self.features=get_features(cnn_weights_path=cnn_weights_path)
		self.avg=nn.AdaptiveAvgPool2d(output_size=1)
		self.fc_all=nn.Linear(in_features=channel_size,out_features=classes_num)
		self.drop=nn.Dropout(p=drop_rate)
		self.criterion=nn.CrossEntropyLoss()
		if cnn_weights_path is None:
			self.__init_weights__()

	def forward(self,x):
		x=self.features(x)
		b,c,h,w=x.size()
		x=self.avg(x).view(b,c)
		x=self.drop(x)
		x=self.fc_all(x)
		return x
	def get_attention_map(self,image_path,transform,save_path):
		image=Image.open(image_path).convert('RGB')
		#生成输入variable
		img=transform(image)
		img=img.unsqueeze(0)
		img=Variable(img,requires_grad=False)
		feature_map=self.features(img)
		#生成heatmap map
		heatmap=torch.mean(feature_map,dim=1).squeeze().data.numpy()
		heatmap=cv2.resize(heatmap,(224,224))
		heatmap=(heatmap-heatmap.min())/(heatmap.max()-heatmap.min())
		heatmap=heatmap*255
		#生成结果
		image=cv2.cvtColor(np.asarray(image.resize((224,224))),cv2.COLOR_BGR2RGB)
		color_map = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
		img_res = cv2.addWeighted(image.astype(np.uint8), 0.5, color_map.astype(np.uint8), 0.5, 0)
		cv2.imwrite(save_path,img_res)
	def get_count_result(self,image_path,transform):
		image=Image.open(image_path).convert('RGB')
		#生成输入variable
		img=transform(image)
		img=img.unsqueeze(0)
		img=Variable(img,requires_grad=False)
		feature_map=self.features(img)
		#统计结果
		feature_vec=self.avg(feature_map).squeeze()
		return feature_vec.data.numpy()
	def get_boundbox_image(self,image_path,transform,save_path=None):
		image=Image.open(image_path).convert('RGB')
		#生成输入variable
		img=transform(image)
		img=img.unsqueeze(0)
		img=Variable(img,requires_grad=False)
		feature_map=self.features(img)
		#生成heatmap map
		heatmap=torch.mean(feature_map,dim=1).squeeze().data.numpy()
		heatmap=cv2.resize(heatmap,(224,224))
		heatmap=(heatmap-heatmap.min())/(heatmap.max()-heatmap.min())
		heatmap=heatmap*255
		p1,p2=self.get_boundbox(heatmap)
		if save_path is not None:
			#生成结果
			image=cv2.cvtColor(np.asarray(image.resize((224,224))),cv2.COLOR_BGR2RGB)
			color_map = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
			img_res = cv2.addWeighted(image.astype(np.uint8), 0.5, color_map.astype(np.uint8), 0.5, 0)
			cv2.rectangle(img_res,p1,p2,color=(255,0,0),thickness=2)
			cv2.imwrite(save_path,img_res)
		return p1,p2
	def get_boundbox(self,heatmap):
		heatmap=heatmap>180
		rows=heatmap.sum(axis=0)
		cols=heatmap.sum(axis=1)
		x1,y1,x2,y2=0,0,cols.shape[0],rows.shape[0]
		for i in range(rows.shape[0]):
			if rows[i]!=0:
				x1=i
				break
		for i in range(0,rows.shape[0]):
			if rows[rows.shape[0]-i-1]!=0:
				x2=rows.shape[0]-i-1
				break
		for i in range(cols.shape[0]):
			if cols[i]!=0:
				y1=i
				break
		for i in range(0,cols.shape[0]):
			if cols[cols.shape[0]-i-1]!=0:
				y2=cols.shape[0]-i-1
				break
		return (x1,y1),(x2,y2)
	def __init_weights__(self):
		'''
		用于初始化模型参数
		'''
		for m in self.modules():
			if isinstance(m,nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m,nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.fill_(0)
			elif isinstance(m,nn.Linear):
				m.bias.data.zero_()
		print("model weights init succeed")


def get_net(classes_num=8,channel_size=2048,cnn_weights_path=None,all_weights_path=None,
	drop_rate=0):
	model=Net(classes_num=classes_num,channel_size=channel_size,
		drop_rate=drop_rate,cnn_weights_path=cnn_weights_path)
	if all_weights_path!=None:
		model.load_state_dict(torch.load(all_weights_path))
		print("{} load succeed...".format(all_weights_path))
	return model


if __name__=="__main__":
	pass
