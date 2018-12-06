import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch
class VGG(nn.Module):

	def __init__(self, features, num_classes=1000, init_weights=True):
		super(VGG, self).__init__()
		self.features = features
		if init_weights:
			self._initialize_weights()

	def forward(self, x):
		x = self.features(x)		
		return x

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
	layers = []
	in_channels = 3
	for v in cfg:
		if v == 'M':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]
			in_channels = v
	return nn.Sequential(*layers)


cfg = {
	'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg16():
	"""VGG 16-layer model (configuration "D")

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = VGG(make_layers(cfg['D']))
	return model

def vgg19():
	"""VGG 19-layer model (configuration "E")

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = VGG(make_layers(cfg['E']))
	return model

if __name__=="__main__":
	model=vgg16()
	count=0
	for p in model.parameters():
		count+=p.numel()
	print(count/1e6)

	inputs=torch.autograd.Variable(torch.randn(1,3,448,448))
	outputs=model(inputs)
	print(outputs.shape)
