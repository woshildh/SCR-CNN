import torch,os
from torchvision import transforms
import config,ImageReader,Metrics,sfr
from tensorboardX import SummaryWriter
from torch.optim import SGD
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.backends.cudnn as cudnn
import time

def get_ctime():
	t=time.strftime("%Y-%m-%d %H:%M:%S")
	return t

#设置使用gpu的id
if config.use_cuda:
	os.environ["CUDA_VISIBLE_DEVICES"]=config.gpu_id

def train():
	'''
	进行训练
	'''
	#定义记录部分
	csv_path=config.csv_path
	tb_path=config.tb_path
	writer=SummaryWriter(log_dir=tb_path)

	#定义模型

	if config.start_epoch==1:
		model=sfr.get_sfr(classes_num=config.classes_num,
			channel_size=config.channel_size,
			cnn_weights_path=config.cnn_weights_path,drop_rate=config.drop_rate,
			sr_rate=config.sr_rate,fr_rate=config.fr_rate)
	else:
		model=sfr.get_sfr(classes_num=config.classes_num,
			channel_size=config.channel_size,drop_rate=config.drop_rate,
			sr_rate=config.sr_rate,fr_rate=config.fr_rate)

	print("model load succeed")

	if config.use_cuda:
		model=model.cuda() #将model转到cuda上
	if config.use_parallel:
		model=nn.DataParallel(model,device_ids=config.device_ids)
		cudnn.benchmark=True
	if config.start_epoch!=1:
		all_weights_path=config.save_weights_path.format(config.start_epoch-1)
		model.load_state_dict(torch.load(all_weights_path))
		print("{} load succeed".format(all_weights_path))
	
	#加载数据集
	train_folder=config.train_img_path
	validate_folder=config.validate_img_path
	train_loader=ImageReader.getLoader(config.dataset,"train",
		train_folder)
	validate_loader=ImageReader.getLoader(config.dataset,"validate",
		validate_folder)

	#定义优化器和学习率调度器
	optimizer=SGD(params=model.parameters(),lr=config.start_lr,
		momentum=0.9,weight_decay=config.weight_decay)
	scheduler=optim.lr_scheduler.MultiStepLR(optimizer,milestones=config.stones,
		gamma=0.1)

	#定义评估函数
	criterion=nn.CrossEntropyLoss()
	accuracy=Metrics.Accuracy()

	#定义最好的准确率
	best_acc=0
	for i in range(config.start_epoch,config.start_epoch+config.num_epoch):
		#分配学习率
		scheduler.step(epoch=i)
		lr=scheduler.get_lr()[0]

		print("{} epoch start , lr is {}".format(i,lr))

		#开始训练这一轮
		model.train()
		accuracy.__init__()
		train_loss=0
		train_step=0

		for x,y in train_loader:
			x=Variable(x)
			y=Variable(y)
			if config.use_cuda:
				x=x.cuda()
				y=y.cuda(async=True)
			optimizer.zero_grad() #清空梯度值
			y_=model(x) #求y
			#求这一步的损失值和准确率
			step_loss=criterion(y_,y)
			step_acc=accuracy(y_,y)
			train_loss+=step_loss.data[0]

			#更新梯度值
			step_loss.backward()
			optimizer.step()
			
			train_step+=1  #训练步数+1

			#输出这一步的记录
			print("{} epoch,{} step,step loss is {:.6f},step acc is {:.4f}".format(
				i,train_step,step_loss.data[0],step_acc))
			del(step_loss,x,y,y_)
		#求这一轮训练情况
		train_acc=accuracy.total_correct/(accuracy.total_sample+1e-5)
		train_loss=train_loss/(train_step+1e-5)

		#保存模型
		weights_name=config.save_weights_path.format(i)
		torch.save(model.state_dict(),weights_name)
		del_weights_name=config.save_weights_path.format(i-3)
		if os.path.exists(del_weights_name):
			os.remove(del_weights_name)
		print("{} save,{} delete".format(weights_name,del_weights_name))

		#开始验证步骤
		model.eval()
		accuracy.__init__()  #将accuracy中total_sample和total_correct清0
		val_loss=0
		val_step=0
		for x,y in validate_loader:
			x=Variable(x,requires_grad=False)
			y=Variable(y,requires_grad=False)
			if config.use_cuda:
				x=x.cuda()
				y=y.cuda(async=True)
			y_=model.forward_validate(x)
			step_loss=criterion(y_,y)
			step_acc=accuracy(y_,y)
			val_loss+=step_loss.data[0]
			val_step+=1
			del(x,y,y_,step_loss)
		val_acc=accuracy.total_correct/(accuracy.total_sample+1e-5)
		val_loss=val_loss/(val_step+1e-5)
		print("validate end,log start")
		
		#保存最佳的模型
		if best_acc<val_acc:
			weights_name=config.save_weights_path.format("best_acc")
			torch.save(model.state_dict(),weights_name)
			best_acc=val_acc

		#求model的正则化项
		l2_reg=0.0
		for param in model.parameters():
			l2_reg += torch.norm(param).data[0]
		#开始记录
		with open(csv_path,"a",encoding="utf-8") as file:
			t=get_ctime()
			content="{},{:.6f},{:.4f},{:.6f},{:.4f},{:.6f},{}".format(i,train_loss,
				train_acc,val_loss,val_acc,l2_reg,t)+"\n"
			file.write(content)
		writer.add_scalar("Train/acc",train_acc,i)
		writer.add_scalar("Train/loss",train_loss,i)
		writer.add_scalar("Val/acc",val_acc,i)
		writer.add_scalar("Val/loss",val_loss,i)
		writer.add_scalar("lr",lr,i)
		writer.add_scalar("l2_reg",l2_reg,i)
		print("log end ...")


		print("{} epoch end, train loss is {:.6f},train acc is {:.4f},val loss is \
{:.6f},val acc is {:.4f},weight l2 norm is {:.6f}".format(i,train_loss,train_acc,val_loss,val_acc,l2_reg))
	del(model)
	print("{} train end,best_acc is {}...".
		format(config.dataset,best_acc))

if __name__=="__main__":
	train()

