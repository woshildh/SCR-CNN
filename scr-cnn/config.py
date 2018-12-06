#定义cuda和gpu相关
use_cuda=False
use_parallel=False
gpu_id="1"
device_ids=[1]
#定义数据集的相关部分
dataset="cub"
train_img_path="../data/cub/CUB200/images/train"
validate_img_path="../data/cub/CUB200/images/val"

#定义网络参数
net_name="resnet50"
classes_num=200
channel_size=2048
drop_rate=0

sr_rate=[1,1,1]
fr_rate=[0.8,0.8,1.2]

#定义数据处理部分
data_aug=True
target_size=224
num_workers=1
data_mean=(0.485, 0.456, 0.406)         
#cifar10:(0.4914, 0.4822, 0.4465)  cifar100:(0.5071, 0.4867, 0.4408)
#cub:(0.4844,0.4985,0.4278)
data_std=(0.229, 0.224, 0.225)    
#cifar10:(0.247032, 0.243485, 0.261587)  cifar100:(0.2675, 0.2565, 0.2761)
#cub:(0.1748,0.1732,0.1841)

#定义记录的路径
csv_path="./logs/cub/csvlog/sfr_{}_cub.csv".format(net_name)
tb_path="./logs/cub/tblog/sfr_{}_cub/".format(net_name)

#定义训练相关的超参数
start_epoch=1
num_epoch=200-start_epoch
batch_size=32
start_lr=0.01
weight_decay=1e-4
stones=[100,150]

#定义权重保存
cnn_weights_path="./weights/pretrained/resnet50.pth"
save_weights_path="./weights/cub/sfr_"+net_name+"_cub_{}.pth"

