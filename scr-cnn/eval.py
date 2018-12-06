import sfr,ImageReader,config,Metrics
import torch
from torch.autograd import Variable
import torch.nn as nn

model=sfr.get_sfr(classes_num=config.classes_num,
	channel_size=config.channel_size,drop_rate=config.drop_rate,
	sr_rate=config.sr_rate,fr_rate=config.fr_rate)
validate_loader=ImageReader.getLoader(config.dataset,"validate",
		config.validate_img_path)
all_weights_path="./weights/cub/sfr_resnet50_cub_best_acc.pth"
if config.use_cuda:
	model=model.cuda()
model.load_state_dict(torch.load(all_weights_path))
model.eval()
val_loss=0
val_step=0
accuracy=Metrics.Accuracy()
criterion=nn.CrossEntropyLoss()

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
print("Validate acc is {}, validate loss is {}".format(val_acc,val_loss))
