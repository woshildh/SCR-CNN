import torch
import config
from torchvision import transforms
from  torchvision.datasets import ImageFolder,CIFAR10,CIFAR100
from torch.utils.data import DataLoader
from torchvision.utils import save_image,make_grid
import torchvision


#定义数据转换的相关操作
if config.data_aug==True:
	if "cifar" in config.dataset:
		t=[transforms.Resize(size=config.target_size),
		   transforms.RandomCrop(size=config.target_size,padding=config.target_size[0]//8),
		   transforms.RandomHorizontalFlip(p=0.5),
		   transforms.ToTensor(),
		   transforms.Normalize(config.data_mean,config.data_std)]
	elif "cub" in config.dataset:
		t=[transforms.Resize(256),
			transforms.RandomCrop(config.target_size),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(config.data_mean, config.data_std)]
	train_transform=transforms.Compose(t)
else:
	train_transform=transforms.Compose([
			transforms.Resize(size=config.target_size),
			transforms.ToTensor(),
			transforms.Normalize(config.data_mean,config.data_std)
		])

if "cifar" in config.dataset:
	validate_transform=transforms.Compose([
			transforms.Resize(size=config.target_size),
			transforms.ToTensor(),
			transforms.Normalize(config.data_mean,config.data_std)])
elif "cub" in config.dataset:
	validate_transform=transforms.Compose([
		transforms.Resize(256),
        transforms.CenterCrop(config.target_size),
        transforms.ToTensor(),
        transforms.Normalize(config.data_mean, config.data_std)])

ms_transform=transforms.Compose([
			transforms.Resize(size=config.target_size),
			transforms.ToTensor()
	])

def get_cub_loader(mode,folder_path):
	'''
	获取loader用于加载数据
	params:
		mode:"train" or "validate"
		folder_path:image folder path
	'''
	if mode=="train":
		dataset=ImageFolder(folder_path,transform=train_transform)
		data_loader=DataLoader(dataset,batch_size=config.batch_size,
			shuffle=True,num_workers=config.num_workers)
	
	elif mode=="validate":
		dataset=ImageFolder(folder_path,transform=validate_transform)
		data_loader=DataLoader(dataset,batch_size=config.batch_size,
			shuffle=False,num_workers=config.num_workers)
	
	else:
		raise ValueError("get_loader mode is error")

	return data_loader

def get_cifar10_loader(mode,root_path):
	'''
	获取cifar10 loader
	'''
	if mode=="train":
		dst=CIFAR10(root=root_path,train=True,transform=train_transform,
			download=False)
		data_loader=DataLoader(dst,batch_size=config.batch_size,shuffle=True,
			num_workers=config.num_workers)
	elif mode=="validate":
		dst=CIFAR10(root=root_path,train=False,transform=validate_transform,
			download=False)
		data_loader=DataLoader(dst,batch_size=config.batch_size,shuffle=False,
			num_workers=config.num_workers)
	else:
		raise ValueError("get_loader mode is error")
	return data_loader
def get_cifar100_loader(mode,root_path):
	'''
	获取cifar10 loader
	'''
	if mode=="train":
		dst=CIFAR100(root=root_path,train=True,transform=train_transform,
			download=False)
		data_loader=DataLoader(dst,batch_size=config.batch_size,shuffle=True,
			num_workers=config.num_workers)
	elif mode=="validate":
		dst=CIFAR100(root=root_path,train=False,transform=validate_transform,
			download=False)
		data_loader=DataLoader(dst,batch_size=config.batch_size,shuffle=False,
			num_workers=config.num_workers)
	else:
		raise ValueError("get_loader mode is error")
	return data_loader

def getLoader(name,mode,folder_path):
	if name=="cub":
		data_loader=get_cub_loader(mode,folder_path)
	elif name=="cifar100":
		data_loader=get_cifar100_loader(mode,folder_path)
	elif name=="cifar10":
		data_loader=get_cifar10_loader(mode,folder_path)
	else:
		raise ValueError("name is error")
	return data_loader
def get_mean_std(dst):
	'''
	根据dataset求得数据的mean和std
	'''
	loader=DataLoader(dst,batch_size=1,num_workers=1)
	mean = torch.zeros(3)
	std = torch.zeros(3)
	for x,y in loader:
		for i in range(3):
			mean[i]+=x[:,i,:,:].mean()
			std[i]+=x[:,i,:,:].std()
	mean.div_(len(loader))
	std.div_(len(loader))
	return mean,std
	
if __name__=="__main__":
	
	dst=dataset=ImageFolder("../data/cub/CUB200/images/val/",transform=ms_transform)
	m,s=get_mean_std(dst)
	print(m,s)

