import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
import numpy as np
import os
import copy
import random
import pandas
import matplotlib.pyplot as plt
from torch.autograd import Function

from G02_dataset import myDataset 
from G01_parameters import G_Parameter, MolPara, ConvertTensor
from A01_util import *

#class MyLoss(Function):
#	@staticmethod
#	def forward(ctx,Y,ref):
#		ctx.save_for_backward(Y, ref)
#		loss = (Y-ref).pow(2).mean()
#		return loss
#	@staticmethod
#	def backward(ctx,grad_output):
#		Y,ref = ctx.saved_tensors
#		n     = Y.shape[0]
#		grad_Y = 2 * (Y - ref) / n
#		grad_ref = None 
#		return grad_Y, grad_ref
#
#class MyLoss2(nn.Module):
#	def __init__(self,ref):
#		super(MyLoss2,self).__init__()
#		self.ref = ref
#	def forward(self,gen_CTable):
#		loss = ((gen_CTable - self.ref) ** 2).mean()
#		return loss
#

def check_grad(tensor):
	print("gradient availability check ",tensor.requires_grad)

class CustomLoss(G_Parameter):
	def __init__(self):
		super().__init__()
	# use ctable
	def ctable_loss(self,gen_tensor):
		ref_CRow_T = self.ref_CRow_T
		x,ThreeN   = gen_tensor.shape
		NAtom      = int(ThreeN/3)

		gen_CRow = Calc_CRow_T(NAtom, gen_tensor)

		diff       = ref_CRow_T - gen_CRow
		diff       = torch.linalg.norm(diff)
		return diff

	def batch_ctable_loss(self,batch_tensor):
		ref_CRow_T = self.ref_CRow_T
		
		diff = 0
		batch_size,ThreeN = batch_tensor.shape
		NAtom = int(ThreeN/3)
		for t in range(batch_size):
			gen_CRow = Calc_CRow_T(NAtom, batch_tensor[t])

			delta  = ref_CRow_T - gen_CRow
			delta  = torch.linalg.norm(delta)
#			check_grad(delta)
			diff  += delta
		return diff




if __name__=="__main__":

	args = ParseInp()
	molinfo   = MolPara()
	Gpara     = G_Parameter() 

	myloss    = CustomLoss()

	threeN    = myloss.ThreeN
	tmp = np.random.rand(5,threeN)
	tmp = ConvertTensor(tmp)
	diff = myloss.batch_ctable_loss(tmp)
#	input_tensor = torch.rand(3,2)
	check_grad(diff)

