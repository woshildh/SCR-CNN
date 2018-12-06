'''
定义准确率计算和混淆矩阵计算这部分
'''

import torch
import torch.nn as nn
import torch.nn.functional  as F

class Accuracy(nn.Module):
	def __init__(self):
		super(Accuracy,self).__init__()
		self.total_sample=0
		self.total_correct=0

	def forward(self,logits,labels):
		'''
		params:
			logits:预测的score,shape is [batch_size,classes_num]
			labels:标签,shape is [batch_size], LongTensor
		return:
			acc:float
		'''
		max_value,max_pos=torch.max(logits,dim=1)
		correct_num=torch.eq(max_pos,labels).sum().data[0]
		batch_size=labels.size(0)

		self.total_sample+=batch_size
		self.total_correct+=correct_num

		acc=correct_num/batch_size

		return acc

