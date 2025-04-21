import os
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import pandas
from torch.nn import BatchNorm1d,LayerNorm

from G02_dataset import f02_main
from G01_parameters import f01_main,ConvertTensor
from A01_util import check_grad

def weight_init(m):
	if isinstance(m, nn.Linear):
		nn.init.xavier_normal_(m.weight)
		nn.init.constant_(m.bias, 0)
	elif isinstance(m, nn.Conv2d):
		pass


class Discriminator(nn.Module):
	""" Architecture of the Generator, uses res-blocks """

	def __init__(self,para):
		super().__init__()
		self.info = para

		self.model           = self.D_model()
#		self.model.apply(weight_init)

		self.optimiser       = para.optimizer(self.parameters(),lr=self.info.lr)
	
		self.counter  = 0
		self.progress = []
		print("Discriminator model")	
		print(self.model)
	def D_model(self):
		info            = self.info
		model_flag      = info.model_flag
		Nlayer          = info.Nlayer
		NAtom           = info.NAtom
		increment       = info.increment
		activation_func = info.activation_func
		batch_size      = info.batch_size
		inp_dim         = NAtom*3
		Normalizer      = self.info.Normalization


		if model_flag==1:

			n1 = inp_dim
			n2 = inp_dim - increment
			n3 = inp_dim - increment*3

			model = nn.Sequential(
		                 nn.Linear(n1, n2),
	           		     activation_func,
	           			 nn.Linear(n2, n3),
						 activation_func,
						 nn.Linear(n3 ,1),
						 nn.Sigmoid()
	        )

		if model_flag==102:
			module_list=[]
			for i in range(Nlayer):
				module_list.append(nn.Linear(inp_dim,inp_dim+increment))
				module_list.append(activation_func)
				module_list.append(Normalizer(inp_dim+increment))
				inp_dim+=increment
	
			for i in range(Nlayer):
				module_list.append(nn.Linear(inp_dim,inp_dim-increment))
				module_list.append(activation_func)
				module_list.append(Normalizer(inp_dim-increment))
				inp_dim-=increment

			module_list.append(nn.Linear(inp_dim,1))
			module_list.append(activation_func)
	
			model = nn.Sequential(*module_list)


		if model_flag==2:
				module_list=[]
				for i in range(Nlayer):
					module_list.append(nn.Linear(inp_dim,inp_dim+increment))
					module_list.append(activation_func)
					inp_dim+=increment
		
				for i in range(Nlayer):
					module_list.append(nn.Linear(inp_dim,inp_dim-increment))
					module_list.append(activation_func)
					inp_dim-=increment
	
				module_list.append(nn.Linear(inp_dim,1))
				module_list.append(activation_func)
		
				model = nn.Sequential(*module_list)
	
		return model



	def forward(self, inp_tensor):
		out=self.model(inp_tensor)
		return out

	def train(self, inputs, flag):
	
		#compute the NN output
		outputs = self.forward(inputs)
	
		#compute the loss
		if flag==1:
			targets = torch.ones_like(outputs)
		elif flag==0:
			targets = torch.zeros_like(outputs)
		loss    = self.info.loss_func(outputs,targets)


		#increase counter and accumulate error every 10
		self.counter += 1
		if self.counter%10 == 0:
			self.progress.append(loss.item())

		if self.counter%10000 ==0:
			print("counter = ", self.counter)

		#  perform a backward pass, update weights
		self.optimiser.zero_grad()
		loss.backward()
		self.optimiser.step()


	def plot_progress(self):
		df = pandas.DataFrame(self.progress,columns=['loss'])
		df.plot(marker='.',grid=True,title="discriminator")
		plt.savefig("discriminator.png")
		plt.show()

def generate_random(size):
	random_data = torch.rand(size)
	random_data.requires_grad_(True)	
	return random_data

if __name__=="__main__":

	folder = "Al12_generated_structures" 

	g_p, d_p,arg       = f01_main()
	dataloader,dataset = f02_main(folder)	
	NAtom              = dataset.configs[0].NAtom
	g_p.NAtom          = NAtom
	d_p.NAtom          = NAtom

	discriminator = Discriminator(d_p)	

#----
	real_flag=1
	fake_flag=0
	for i in range(1000):
		for idx,geom_c in enumerate(dataloader):
#			real_label = torch.ones_like(geom_c)
#			real_label = torch.FloatTensor([1.0])
			discriminator.train(geom_c,real_flag)
			fake_sample = generate_random(NAtom*3)
#			fake_label  = torch.zeros_like(fake_sample)
#			fake_label  = torch.FloatTensor([0.0])
			discriminator.train(fake_sample,fake_flag)
	

	discriminator.plot_progress()

#	fake_sample = generate_random(NAtom*3)
#	fake_sample_label = discriminator(fake_sample)
#	print(fake_sample_label)
#	check_grad(fake_sample)
#
#	fake_label  = torch.zeros_like(fake_sample)
#	discriminator.train(fake_sample,fake_label)
#
#
#	
#	for idx,geom_zscore in enumerate(dataloader):
#		geom_zscore.requires_grad_(True)
#		real_label = torch.ones_like(geom_zscore)
##		check_grad(geom_zscore)
#		discriminator.train(geom_zscore, real_label)
#	
#
