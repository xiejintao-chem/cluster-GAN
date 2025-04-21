import torch
from torch.autograd import Variable
import os
import numpy as np
import argparse
import torch.nn as nn
import copy
import random

from G02_dataset import f02_main
from G01_parameters import f01_main,ConvertTensor
from A01_util import check_grad

from G20_discriminator import *
from G11_generator_no_punish_class import *
def plot_progress(data,title):
		df = pandas.DataFrame(data,columns=['loss'])
		df.plot(marker='.',grid=True,title=title)
		plt.savefig("generator_loss.png")
		plt.show()


def Gan_train(generator,discriminator,dataloader):

	Epoch=1000
	discriminator_loss=[]
	generator_loss=[]

	thresh1 = generator.info.thresh1
	thresh2 = generator.info.thresh2

	random_inp  = generator.Gen_Random_input()
	for epoch in range(Epoch):
		for i,real_geom in enumerate(dataloader):

			#train generator
#			print("train generator")
			g_loss,gen_tensor = generator.train(discriminator,target_flag=1,maxit=1000)


			#train discriminator
			
			#1)train discriminator by feeding real data
			d_res = discriminator(real_geom)
			real_label = torch.ones_like(d_res)
			real_loss  = discriminator.info.loss_func(d_res,real_label)

			#2) train discriminator by feeding fake data
			discriminator.optimiser.zero_grad()
			d_res      = discriminator(gen_tensor.detach())
			fake_label = torch.zeros_like(d_res) #should I generate d_res again?
			fake_loss  = discriminator.info.loss_func(d_res,fake_label)
 
		
			d_loss = (real_loss + fake_loss)*0.5

			d_loss.backward()
			discriminator.optimiser.step() 
	
			discriminator_loss.append(d_loss.item())
			generator_loss.append(g_loss.item())
#			if epoch%20==0:		
			print(
   	         "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
   	         % (epoch, Epoch, i, len(dataloader), d_loss.item(), g_loss.item()))
	
	#	if abs(g_loss.item()-0.25) < 0.1 and abs(d_loss.item()-0.25) < 0.1:
	#		print("break")
	#		break			

	plot_progress(discriminator_loss,"d_loss")
	plot_progress(generator_loss,"g_loss")
	return generator,gen_tensor

if __name__=="__main__":

	folder = "Al12_generated_structures" 

	g_p, d_p           = f01_main()
	dataloader,dataset = f02_main(folder)	
	NAtom              = dataset.configs[0].NAtom

	g_p.NAtom          = NAtom
	d_p.NAtom          = NAtom
	g_p.Gflag_network  = 2 # =1, use network, =2 not use network

	g_p.configs        = dataset.configs
	discriminator = Discriminator(d_p)	
	generator = Generator(g_p)


	generator,gen_tensor = Gan_train(generator,discriminator,dataloader)

	Gen_XYZ(gen_tensor,dataset.configs[0].atoms)
	torch.save(generator,"g31_generator.pth")

