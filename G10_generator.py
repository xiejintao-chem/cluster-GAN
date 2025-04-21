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
from G20_discriminator import Discriminator

#from G03_myloss import CustomLoss
from A01_util import *

class DistancePenalty(nn.Module):
	def __init__(self,thresh_min, thresh_max):
		super().__init__()
		self.thresh_min = thresh_min
		self.thresh_max = thresh_max
		self.w1         = 1
		self.w2         = 1

	def forward(self,flatten_geom):
		#geom = flatten_geom.view(-1,3)
		#pairwise_dist = torch.cdist(geom,geom)
		pairwise_dist = self.Calc_BondLength(flatten_geom)

		below_thresh_penalty = torch.relu (self.thresh_min - pairwise_dist)
		above_thresh_penalty = torch.relu (pairwise_dist   - self.thresh_max)

		punish_below = below_thresh_penalty.mean()
		punish_above = above_thresh_penalty.mean()
#		print(self.thresh_min - pairwise_dist,"bwlo")
#		print(below_thresh_penalty,"below")
		return self.w1*punish_below + self.w2*punish_above

	
	def Calc_BondLength(self,geom):
		thresh1 = self.thresh_min #smaller
		thresh2 = self.thresh_max #larger
		w1      = self.w1
		w2      = self.w2
	
		ThreeN = len(geom)
		NAtom  = int(ThreeN/3)
		geom = geom.view(NAtom,3)
		bond_length=[]
		for i in range(NAtom):
			for j in range(i+1,NAtom):
				diff = geom[i,:]-geom[j,:]
				diff = torch.linalg.norm(diff)
				bond_length.append(diff)

		return ConvertTensor(bond_length)



def weight_init(m):
	if isinstance(m, nn.Linear):
#		nn.init.xavier_normal_(m.weight)
#		nn.init.constant_(m.weight,1)
		nn.init.normal_(m.weight,mean=0,std=0.1)
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

		self.thresh1 = self.info.thresh1
		self.thresh2 = self.info.thresh2

		self.penalty = DistancePenalty(self.thresh1,self.thresh2)
		#---------------------------------------------
		#the below are used for punishment function
		#for flag 1
		self.RandomInput  = self.Gen_Random_input()

		#for flag 2
		self.rand_std     = self.Gen_Random_distribution()
		print("max and min std")
		print(self.max_std)
		print(self.min_std)

	
		#--------------------------------------------
		#self.model.apply(weight_init)
		print("Generator model")
		print(self.model)

	###################################################
	def G_model(self):
		info       = self.info

		model_flag = info.model_flag
	#	print("model flag",model_flag)
		Nlayer     = info.Nlayer
		NAtom      = info.NAtom
		increment  = info.increment
		activation_func = info.activation_func
		inp_dim         = NAtom*3 
		Normalizer      = self.info.Normalization
	#-----------------------------------------------
		if model_flag ==1: #simple
			model = nn.Sequential(
		                 nn.Linear(inp_dim, inp_dim),
	           		     activation_func
	        )

		if model_flag ==101: #simple
	#		print("lalalalall")
			model = nn.Sequential(
		                 nn.Linear(inp_dim, inp_dim),
	           		     activation_func,
						 Normalizer(inp_dim)
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

	#--------------------------------------------------
		if model_flag==102:#increase then decrease
			module_list=[]
			for i in range(Nlayer):
				module_list.append(nn.Linear(inp_dim,inp_dim+increment))
				module_list.append(Normalizer(inp_dim+increment))
				module_list.append(activation_func)
				inp_dim+=increment
	
			for i in range(Nlayer):
				module_list.append(nn.Linear(inp_dim,inp_dim-increment))
				module_list.append(Normalizer(inp_dim-increment))
				module_list.append(activation_func)
				inp_dim-=increment

			module_list.append(nn.Linear(inp_dim,inp_dim))	
			model = nn.Sequential(*module_list)


		return model
	#-----------------------------------------------
	#######################################################


	def forward(self,input_tensor,weight,bias=0):

		out=self.model(input_tensor)
		out=torch.matmul(out,weight)+bias
		out=out+bias
#		print(out.shape)
	#	out = out.view(1,-1)
		return out


	#generate random distribution with std between max_std and min_std
	def Gen_Random_distribution(self):
		self.ceil_bond_length = np.ceil(self.info.avg_bond_length)  #estimated bond length,defined in G01
		self.max_std          = 3 #np.std( np.arange(self.info.NAtom-1)*self.ceil_bond_length )# max distribution corresponding to the linear configuration	
		self.min_std          = 2 #LQNOTE, currently manually set 2, just for testing purpose. May need better rationalization later. 
		max_std = self.max_std
		min_std = self.min_std
		rand    = torch.rand(1)
		rand    = min_std + (max_std-min_std)*rand
		return rand


	def Gen_Random_input(self):
		NAtom = self.info.NAtom
		mean  = self.info.configs[0].mean
		std   = self.info.configs[0].std
		inp   = torch.randn(NAtom*3)
		input_tensor = inp*std + mean
		input_tensor.requires_grad_(True)	
		return input_tensor
	
	def train(self,discriminator,target_flag,maxit=1000):

		thresh1 = self.info.thresh1
		thresh2 = self.info.thresh2
		input_tensor = self.Gen_Random_input()
		init_bias = torch.zeros_like(input_tensor)
		init_weight = torch.ones(self.info.NAtom*3,self.info.NAtom*3)	
		self.w1=1
		self.w2=1

		gen_tensor = input_tensor
		for i in range(maxit):
		
			#generate fake data	
			gen_tensor   = self.forward(gen_tensor.detach(),init_weight,init_bias)
			self.gen_tensor = gen_tensor
			#calculate the punishment of the fake data
			
			#punish  = Punishment(gen_tensor.detach(),self,flag=1)	
			#punish  = torch.max(punish,torch.zeros_like(punish))
			punish   = self.penalty(gen_tensor.detach())

			print(punish)	
			if punish.item() - 0.001 <0:
				break
		#	assert punish.detach()>10, print("punish explode",punish)
				
			#run the discriminator
			d_output = discriminator.forward(gen_tensor) #lqattempt1

			if target_flag==1:
				targets=torch.ones_like(d_output)
			elif target_flag==0:
				targets=torch.zeros_like(d_output)

			#minimize the diff between d_output and the targets
			g_loss = self.info.loss_func(d_output,targets) #  lqattempt2
			loss   = g_loss+punish
			#loss    = discriminator.info.loss_func(punish,torch.zeros_like(punish))


			#get the network parameters
			params = self.state_dict()
			keys = list(params.keys())
			last_b = copy.deepcopy(params[keys[-1]])
			last_w = copy.deepcopy(params[keys[-2]])
			#do some check
			assert len(last_b) == self.info.NAtom*3,	print("error length of last_b not equal to threeN, len(keys)=",len(last_b))
			assert len(last_w) == self.info.NAtom*3,    print("error length of last_w not equal to threeN, len(keys)=",len(last_b))

			#update the init_bias and init_weight
			init_bias = ConvertTensor(last_b,True)
			init_weight = ConvertTensor(last_w,True)	

			#run the backward training
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()


			#plot stuff
			self.info.counter += 1
			if self.info.counter%10==0:
				self.info.progress.append(loss.item())
		#		print(gen_tensor)


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
	
	discriminator = Discriminator(d_p)	

	generator = Generator(g_p)

	init_weight = torch.ones([NAtom*3, NAtom*3])
	geom = generator.forward(generator.RandomInput,weight=init_weight)	
	maxit=10000
	print("max iteration=",maxit)
	loss,gen_tensor = generator.train(discriminator,target_flag=1,maxit=maxit)
	
	generator.plot_progress()

	Gen_XYZ(gen_tensor,dataset.configs[0].atoms)
	params = generator.state_dict()


	torch.save(generator,"g10_generator.pth")

#	for key,value in params.items():
#		print(key)
#	keys = list(params.keys())
#	print(keys)
#	print(type(keys))
#	print(keys[-2])
#	print(params[keys[-2]])
#	print(params[keys[-2]].shape)
