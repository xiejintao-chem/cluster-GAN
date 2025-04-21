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
#from collections import Counter

from G02_dataset import f02_main
from G01_parameters import f01_main,ConvertTensor
from G20_discriminator import Discriminator
from G04_penalty import DistancePenalty


#from G03_myloss import CustomLoss
from A01_util import *
from A02_aux import *
#this is stem from G10_generator. The difference is to rewrite the random input
#so that the random input is generated in every iteration, and the last_w and last_b is no longer used
#this implementation should be more close to the orgianal gan 

#from ~/ML-HessianFigure/GAN/myInGAN/backup/InGAN.py
class LRPolicy(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __call__(self, citer):
        return 1. - max(0., float(citer - self.start) / float(self.end - self.start))

class Generator(nn.Module):
	""" Architecture of the Generator, uses res-blocks """

	def __init__(self,para):
		super().__init__()
		
		self.info = para
		self.info.counter = 0
		self.info.progress = []

		self.maxit  = self.info.maxit
		self.model  = self.G_model()
		self.optimizer = para.optimizer(self.parameters(),lr=para.lr)
		

		#lr scheduler
		#First define linearly decaying functions 
	#	start_decay = 20
	#	end_decay   = para.Epoch
	#	lr_function = LRPolicy(start_decay, end_decay)
	#	self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_function)	


		self.penalty_flag = para.penalty_flag
		self.penalty = DistancePenalty(para,flag=self.penalty_flag)

		#---------------------------------------------
		#the below are used for punishment function
		#for flag 1
		self.RandomInput  = self.Gen_Random_input()

		#for flag 2
		self.rand_std     = self.Gen_Random_distribution()
#		print("max and min std")
#		print(self.max_std)
#		print(self.min_std)

	
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
		if self.info.latent_dim == 0:
			inp_dim     = NAtom*3
		else:
			inp_dim     = self.info.latent_dim 
		Normalizer      = self.info.Normalization
		self.inp_dim    = inp_dim
	#-----------------------------------------------
		if model_flag ==1: #simple
			model = nn.Sequential(
		                 nn.Linear(inp_dim, NAtom*3),
	           		     activation_func
	        )

		if model_flag ==101: #simple
	#		print("lalalalall")
			model = nn.Sequential(
		                 nn.Linear(inp_dim, NAtom*3),
	           		     activation_func,
						 Normalizer(NAtom*3)
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

			module_list.append(nn.Linear(inp_dim,NAtom*3))	
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

			module_list.append(nn.Linear(inp_dim,NAtom*3))	
			model = nn.Sequential(*module_list)


		return model
	#-----------------------------------------------
	#######################################################


	def forward(self,input_tensor):

		out=self.model(input_tensor)
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
	#attempt1
		NAtom = self.info.NAtom
		idx   = random.randint(0,len(self.info.configs)-1)
		mean  = self.info.configs[idx].mean
		std   = self.info.configs[idx].std
		inp   = torch.randn(self.inp_dim)
		input_tensor = inp*std + mean
		input_tensor.requires_grad_(True)	

	#attempt2

	
		return input_tensor
	
	def train(self,discriminator,component_NAtom, subset=0,punish=0,maxit=1000,clip_value=0.05):
		flag_param_dict=0
		for i in range(maxit):
		
			#generate fake data	
			#use random input
	#		random_inp = self.Gen_Random_input()
	#		gen_tensor   = self.forward(random_inp.detach())		
	
			#use avg geom
			avg_geom=cal_avg(subset,component_NAtom)
			gen_tensor   = self.forward(avg_geom)


			#calculate the punishment of the fake data
			punish   = self.penalty(gen_tensor)

			#lqtest
		#	punish.optimizer = torch.optim.Adam(self.parameters(), lr=self.info.lr)	
		#	punish.optimizer.zero_grad()
		#	punish.backward()
		#	punish.optimizer.step()	
			

			#print(punish)	
			#run the discriminator
			d_output = discriminator.forward(gen_tensor) #lqattempt1

			targets=torch.ones_like(d_output)

			#minimize the diff between d_output and the targets
			g_loss = self.info.loss_func(d_output,targets) #  lqattempt2
			loss   = g_loss+punish
			#loss    = discriminator.info.loss_func(punish,torch.zeros_like(punish))


			#run the backward training
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

			# Update learning rate scheduler
			#self.lr_scheduler_G.step()

			#-------------------------------------
			#wgan
			#for p in self.parameters():
			#	p.data.clamp_(-clip_value,clip_value)

			#-----------------------------------
	
			#if punish.item()<0.1:
			#	p_backup = copy.deepcopy(self.parameters())
#			print("********")
#			print(self.parameters().data)		


			if punish < 0.001:
				ref_params_dict = self.state_dict()
				flag_param_dict = 1
			if punish > 0.02 and flag_param_dict == 1:
				print("weight avg of g parameters")
				current_para = self.state_dict()

				#get weighted average of parameters
				weighted_params = {}
				for name, param1 in ref_params_dict.items():
					#get para from the current model
					param2 = current_para[name]
					#weighted average
					weighted_param = 0.9 * param1.data + 0.1 * param2.data
					weighted_params[name] = weighted_param
				
				self.load_state_dict(current_para)

			#plot stuff
			self.info.counter += 1
			if self.info.counter%10==0:
				self.info.progress.append(loss.item())
		#		print(gen_tensor)



		return loss,gen_tensor,punish



	def plot_progress(self):
		df = pandas.DataFrame(self.info.progress,columns=['loss'])
#		df.plot(ylim=[0,0.8],marker='.',grid=True,title="generator_loss" )
		df.plot(marker='.',grid=True,title="generator_loss" )
#		plt.savefig("generator_loss.png")
		plt.show()



#-----


if __name__=="__main__":

#	folder = "Al12_generated_structures" 
	folder = "P11p_generated_structures"
	g_p, d_p,args           = f01_main()
	dataloader,dataset = f02_main(folder)	
	NAtom              = dataset.configs[0].NAtom
	g_p.NAtom          = NAtom
	d_p.NAtom          = NAtom
	g_p.configs        = dataset.configs
	
	discriminator = Discriminator(d_p)	

	generator = Generator(g_p)

	maxit=generator.maxit
	print("max iteration=",maxit)
	start_value=30
	Nconfig = random.randint(start_value,len(dataset.configs)) #choose a random number to setup subset
	subset = random.sample(dataset.configs,Nconfig)
	random.shuffle(subset)


	loss,gen_tensor,punish = generator.train(discriminator,subset,maxit=maxit)
	
	generator.plot_progress()


	Gen_XYZ(gen_tensor,dataset.configs[0].atoms)


	torch.save(generator,"g16_test.pth")

#	for key,value in params.items():
#		print(key)
#	keys = list(params.keys())
#	print(keys)
#	print(type(keys))
#	print(keys[-2])
#	print(params[keys[-2]])
#	print(params[keys[-2]].shape)
