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
from G01_parameters import * 
from G03_myloss import CustomLoss, check_grad
from A01_util import *
from G20_discriminator import *

#in this attempt, use the random as input 



def weight_init(m):
	if isinstance(m, nn.Linear):
#		nn.init.xavier_normal_(m.weight)
#		nn.init.constant_(m.weight,1)
		nn.init.normal_(m.weight,mean=0,std=1.5) #0.5
		nn.init.constant_(m.bias, 0)
	elif isinstance(m, nn.Conv2d):
		pass


class Generator(nn.Module):
	""" Architecture of the Generator, uses res-blocks """

	def __init__(self,para):
		super().__init__()
		
		self.info = para
		self.info.counter = 0
		self.info.progress = []

		self.model  = self.G_model()
		self.optimizer = para.optimizer(self.parameters(),lr=para.lr)
	
		self.model.apply(weight_init)


	##################################################3
	def G_model(self):
		info       = self.info

		model_flag = info.model_flag
		Nlayer     = info.Nlayer
		mat_shape  = info.mat_shape
		increment  = info.increment
		activation_func = info.activation_func

		if model_flag==0: #MNIST
			model = nn.Sequential(
    				nn.Linear(100,128,bias=True),
    				nn.LeakyReLU(0.2, inplace=True),
    				nn.Linear(128,256,bias=True),
    				nn.BatchNorm1d(256, 0.8),
					nn.LeakyReLU(0.2, inplace=True),
    				nn.Linear(256,512,bias=True),
    				nn.BatchNorm1d(512, 0.8),					
    				nn.LeakyReLU(0.2, inplace=True),
    				nn.Linear(512,1024,bias=True),
    				nn.BatchNorm1d(1024, 0.8),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Linear(1024,784,bias=True),
					nn.Tanh()
				)
			print("generator",model)
	#-----------------------------------------------
		if model_flag ==1: #simple
			inp_dim = mat_shape[0]*mat_shape[1]
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
			inp_dim = mat_shape[0]*mat_shape[1]
			for i in range(Nlayer):
				module_list.append(nn.Linear(inp_dim,inp_dim+increment))
				module_list.append(activation_func)
				inp_dim+=increment
	
			for i in range(Nlayer):
				module_list.append(nn.Linear(inp_dim,inp_dim-increment))
				module_list.append(activation_func)
				inp_dim-=increment
	
			model = nn.Sequential(*module_list)
	#-----------------------------------------------------
		if model_flag==3: #decrease then increase
			module_list=[]
			inp_dim = mat_shape[0]*mat_shape[1]
			for i in range(Nlayer):
				module_list.append(nn.Linear(inp_dim,inp_dim-increment))
				module_list.append(activation_func)
				inp_dim-=increment
	
			for i in range(Nlayer):
				module_list.append(nn.Linear(inp_dim,inp_dim+increment))
				module_list.append(activation_func)
				inp_dim+=increment
	
			model = nn.Sequential(*module_list)
	#-----------------------------------------------
		if model_flag==4:#increase then decrease without activation func
			module_list=[]
			inp_dim = mat_shape[0]*mat_shape[1]
			for i in range(Nlayer):
				module_list.append(nn.Linear(inp_dim,inp_dim+increment))
				inp_dim+=increment
	
			for i in range(Nlayer):
				module_list.append(nn.Linear(inp_dim,inp_dim-increment))
				inp_dim-=increment
	
			model = nn.Sequential(*module_list)

	#-----------------------------------------------
		if model_flag == 5:  # same number of neurons
			module_list = []
			inp_dim = mat_shape[0]*mat_shape[1]
			
			module_list.append(nn.Linear(inp_dim,info.latent_dim))
			module_list.append(activation_func)			
			for i in range(Nlayer):
				module_list.append(nn.Linear(info.latent_dim,info.latent_dim))
				module_list.append(activation_func)
			module_list.append(nn.Linear (info.latent_dim, inp_dim)  )
			module_list.append(activation_func)	
			model = nn.Sequential(*module_list)

		if model_flag==6: #MLP
			d_in = info.NAtom*3
			d_hidden = 100
			n_layers = 4
			d_out    = d_in

			bias=True

			model = MLP(d_in, n_layers, d_hidden, d_out, activation=nn.ReLU())
			info.MLP_parameters = model.parameters()

		return model
	#######################################################


	def forward(self,inp):
		#use random input
#		input_tensor = ConvertTensor(inp,True) #include grad


		out=self.model(inp)
		
		return out

	
	def train(self,inp_data,targets):

		gen_tensor  = self.forward(inp_data)

		d_output = self.discriminator(gen_tensor)

		loss = self.info.loss_func(d_output,targets) 

		self.info.counter += 1
		if self.info.counter%10==0:
			self.info.progress.append(loss.item())

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()


	def evaluate(self,inp):
		out=self.model(inp)
		out = out.view(-1,3)
		out = out.detach()
		return out

	def plot_progress(self):
		df = pandas.DataFrame(self.info.progress,columns=['loss'])
#		df.plot(ylim=[0,0.8],marker='.',grid=True,title="generator_loss" )
		df.plot(marker='.',grid=True,title="generator_loss" )
		plt.savefig("generator_loss.png")
		plt.show()



#-----


if __name__=="__main__":



	molinfo   = MolPara()
	Gpara     = G_Parameter() 
	generator = Generator(Gpara)

	print(molinfo)
	print(Gpara)
	model = generator.G_model()
	print(model)

	Epoch=5000
	
	mu = 0
	sigma = 4.89	 
	inp_data = np.random.normal(mu,sigma,molinfo.ThreeN)
#	inp_data = np.mat(inp_data)
	inp_data = ConvertTensor(inp_data,True)	
	Dpara     = D_Parameter() 
	discriminator = Discriminator(Dpara)

	generator.discriminator = discriminator


	fake_out = generator.forward(inp_data)
