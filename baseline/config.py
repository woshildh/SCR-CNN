#定义cuda和gpu相关
use_cuda=True
use_parallel=False
gpu_id="0"
device_ids=[0]
#定义数据集的相关部分
dataset="cub"
train_img_path="../data/cub/CUB200/images/train"
validate_img_path="../data/cub/CUB200/images/val"

#定义网络参数
net_name="resnet50"
classes_num=200
channel_size=2048
drop_rate=0  #0或者0.5

#定义数据处理部分
data_aug=True
target_size=(224,224)
num_workers=4
data_mean=(0.485, 0.456, 0.406)
data_std=(0.229, 0.224, 0.225)

#定义记录的路径
csv_path="./logs/cub/csvlog/{}_cub.csv".format(net_name)
tb_path="./logs/cub/tblog/{}_cub/".format(net_name)

#定义训练相关的超参数
start_epoch=1
num_epoch=260-start_epoch
batch_size=64
start_lr=0.01
weight_decay=1e-4
stones=[140,200]

#定义权重保存
cnn_weights_path="./weights/resnet50.pth"
save_weights_path="./weights/cub/"+net_name+"_cub_{}.pth"

