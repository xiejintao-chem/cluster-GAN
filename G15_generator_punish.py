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

from G02_dataset import f02_main
from G01_parameters import f01_main,ConvertTensor

#from G03_myloss import CustomLoss
from A01_util import *

def Calc_bond_length(geom,thresh1,thresh2):
	ThreeN = len(geom)
	NAtom  = int(ThreeN/3)

	geom = geom.view(NAtom,3)
	bond_length=[]
	for i in range(NAtom):
		for j in range(i+1,NAtom):
			diff = geom[i,:]-geom[j,:]
			diff = torch.linalg.norm(diff)
			bond_length.append(diff)

#			if diff < thresh1:
#				return ConvertTensor(9,True)
#			if diff > thresh2:
#				return ConvertTensor(9,True)

	dist_max = np.max(bond_length)
	dist_min = np.min(bond_length)

	if dist_max > thresh2 or dist_min < thresh1:
		return ConvertTensor(9,True) 
	else:
		return ConvertTensor(0,True)

#	return ConvertTensor(0,True)


class Generator(nn.Module):
	""" Architecture of the Generator, uses res-blocks """

	def __init__(self,para):
		super().__init__()
		
		self.info = para
		self.info.counter = 0
		self.info.progress = []

		self.model  = self.G_model()
		self.optimizer = para.optimizer(self.parameters(),lr=para.lr)
		self.flag_network = para.Gflag_network # =1, use network, =2 not use network	

		self.RandomInput  = self.Gen_Random_input()
		#self.model.apply(weight_init)
		print("Generator model")
		print(self.model)

	##################################################3
	def G_model(self):
		info       = self.info

		model_flag = info.model_flag
		Nlayer     = info.Nlayer
		NAtom      = info.NAtom
		increment  = info.increment
		activation_func = info.activation_func
		inp_dim         = NAtom*3 
	#-----------------------------------------------
		if model_flag ==1: #simple
			n1 = inp_dim
			n2 = inp_dim + increment
			n3 = inp_dim + increment*3
			n4 = inp_dim + increment*2
			model = nn.Sequential(
		                 nn.Linear(n1, n2),
	           		     activation_func,
	           		     nn.Linear(n2, n3),
				#		 nn.BatchNorm1d(n3,0.8),
					     activation_func,
						 nn.Linear(n3,n2),
						 activation_func,
				#		 nn.BatchNorm1d(n2,0.8),
						 activation_func,
						 nn.Linear(n2,n1),
					     nn.Tanh()
	        )
	#----------------------------------------------
		if model_flag==2:#increase then decrease
			module_list=[]
			for i in range(Nlayer):
				module_list.append(nn.Linear(inp_dim,inp_dim+increment))
				module_list.append(activation_func)
				inp_dim+=increment
	
			for i in range(Nlayer):
				module_list.append(nn.Linear(inp_dim,inp_dim-increment))
				module_list.append(activation_func)
				inp_dim-=increment

			module_list.append(nn.Linear(inp_dim,inp_dim))	
			model = nn.Sequential(*module_list)
	#-----------------------------------------------------
		if model_flag==4:#no activation func increase then decrease without activation func
			module_list=[]
			for i in range(Nlayer):
				module_list.append(nn.Linear(inp_dim,inp_dim+increment))
				inp_dim+=increment
	
			for i in range(Nlayer):
				module_list.append(nn.Linear(inp_dim,inp_dim-increment))
				inp_dim-=increment
	
			model = nn.Sequential(*module_list)

		return model
	#-----------------------------------------------
	#######################################################


	def forward(self,input_tensor,flag=1):

		#use random input
		if flag==1:
			out=self.model(input_tensor)
			punish =  Calc_bond_length(out,self.info.thresh1,self.info.thresh2)	
			return out,punish
	
		if flag==2: #not use any network
			out = input_tensor
		
		return out


	def Gen_Random_input(self):
		NAtom = self.info.NAtom
		mean  = self.info.configs[0].mean
		std   = self.info.configs[0].std
		inp   = torch.randn(NAtom*3)
		input_tensor = inp*std + mean
		input_tensor.requires_grad_(True)	
		return input_tensor
	
	def train(self,maxit=1000):

		input_tensor = self.Gen_Random_input()	
		for i in range(maxit):

			gen_tensor,punish   = self.forward(input_tensor, self.flag_network)
	
#			print(gen_tensor)

	#		punish = Calc_bond_length(input_tensor)	
			
	#		targets = torch.ones_like(gen_tensor)
	#		loss    = self.info.loss_func(gen_tensor,targets)
	#		loss    = torch.norm(gen_tensor)
	#		punish  = torch.max(punish,torch.zeros_like(punish))
			loss    = punish#.mean()
				
			self.info.counter += 1
			if self.info.counter%10==0:
				self.info.progress.append(loss.item())
				print(gen_tensor)
	
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

#			if loss.item()<2:
#				print("break")
#				break


		return loss,gen_tensor


	def evaluate(self,inp):
		input_tensor = ConvertTensor(inp)
		out=self.model(input_tensor)
		out = out.view(-1,3)
		out = out.detach()
		return out

	def plot_progress(self):
		df = pandas.DataFrame(self.info.progress,columns=['loss'])
#		df.plot(ylim=[0,0.8],marker='.',grid=True,title="generator_loss" )
		df.plot(marker='.',grid=True,title="generator_loss" )
#		plt.savefig("generator_loss.png")
		plt.show()



#-----


if __name__=="__main__":

	folder = "Al12_generated_structures" 

	g_p, d_p           = f01_main()
	dataloader,dataset = f02_main(folder)	
	NAtom              = dataset.configs[0].NAtom
	g_p.NAtom          = NAtom
	d_p.NAtom          = NAtom
	g_p.configs        = dataset.configs
	g_p.Gflag_network  = 1 # =1, use network, =2 not use network

	generator = Generator(g_p)
	geom = generator.forward(generator.RandomInput)	
#	for i in range(20000):
	loss,gen_tensor = generator.train(maxit=100)
	generator.plot_progress()
	
#	gen_tensor = generator.forward()
	Gen_XYZ(gen_tensor,dataset.configs[0].atoms)
	
