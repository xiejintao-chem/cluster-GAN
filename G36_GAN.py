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
from G16_generator import *
from G05_input_transformation import *
from G04_penalty import DistancePenalty
from A02_aux import *
def plot_progress(data,title):
		df = pandas.DataFrame(data,columns=['loss'])
		df.plot(marker='.',grid=True,title=title)
		plt.savefig("generator_loss.png")
		plt.show()


def Gan_train(generator,discriminator,dataset,component_NAtom=0):

	Epoch=generator.info.Epoch
	discriminator_loss=[]
	generator_loss=[]

	thresh1 = generator.info.thresh1
	thresh2 = generator.info.thresh2

	random_inp  = generator.Gen_Random_input()

	#load model
#	generator = torch.load("g16_generator.pth")
	maxit     = generator.maxit

# clearly the clip value influence the std. larger clip value gives larger std	
	clip_value=800
	start_value=len(dataset.configs)	
	for epoch in range(Epoch):
		Nconfig = random.randint(start_value,len(dataset.configs)) #choose a random number to setup subset
		subset = random.sample(dataset.configs,Nconfig)
		random.shuffle(subset)
		i=0
		for config in subset:

			#train generator  #clip_value=2, g_loss flucturate not satisfactory
			c_value = 0.1-epoch/clip_value
			if c_value < 0.01:
				c_value = 0.01
			g_loss,gen_tensor,punish = generator.train(discriminator,component_NAtom, subset=subset,maxit=maxit,clip_value= c_value )

			#train discriminator
			
			#1)train discriminator by feeding real data
			discriminator.optimiser.zero_grad()
			if component_NAtom==0:
				_real_input_ = config.geom_c 
			else:
				_real_input_ = config.bead_center_geom

			real_geom = ConvertTensor(_real_input_,False)
			d_res = discriminator(real_geom)
			real_label = torch.ones_like(d_res)
			real_loss  = discriminator.info.loss_func(d_res,real_label)

			#2) train discriminator by feeding fake data
			d_res      = discriminator(gen_tensor.detach())
			fake_label = torch.zeros_like(d_res) #should I generate d_res again?
			fake_loss  = discriminator.info.loss_func(d_res,fake_label)
 
		
			d_loss = (real_loss + fake_loss)*0.5

			d_loss.backward()
			discriminator.optimiser.step() 

			#-------------------------------------
			#wgan
			for p in discriminator.parameters():
				p.data.clamp_(-c_value,c_value)
			#-----------------------------------
	
			discriminator_loss.append(d_loss.item())
			generator_loss.append(g_loss.item())
			i+=1
			print(
   		         "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]  [punish: %f]"
   	    	     % (epoch, Epoch, i,len(subset), d_loss.item(), g_loss.item(), punish))


#	plot_progress(discriminator_loss,"d_loss")
#	plot_progress(generator_loss,"g_loss")
	return generator,gen_tensor

if __name__=="__main__":

	case=1
	component_NAtom=0 #the natom for each molecule in the cluster
	if case==1:
		folder = "Al12_generated_structures" 
		pth_file = "g36_Al12.pth"
		gen_folder = "Al12_GAN_trained_structures"
	if case==2:
		folder = "P11p_generated_structures"
		pth_file = "g36_P11.pth"
		gen_folder = "P11_GAN_structures"
	if case==3:
		folder="Mg10_generated_structures"
		pth_file = "g36_Mg10.pth"
		gen_folder = "Mg10_Gan_structures"
	if case==4:
		folder="Na10_generated_structures"
		pth_file = "g36_Na10.pth"
		gen_folder = "Na10_Gan_structures"
	if case==5:
		folder="Al4Si7_generated_structures"
		pth_file = "g36_Al4Si7_att5.pth"
		gen_folder = "Al4Si7_Gan_structures_att5"

	if case==6:
		folder = "subset108_4_GAN"
		pth_file = "g36_Org108.pth"
		gen_folder = "Org108_Gan_structures"

	if case==7:
		folder="C20_conformers"
		pth_file = "g36_c20.pth"
		gen_folder = "C20_Gan_structures"	

	if case==8:
		folder="h2o_clusters"
		pth_file="g36_water100.pth"
		gen_folder="Water100_Gan_structure"
		component_NAtom=3
		cond_flag=3

	if case==9:
		folder="h2o_clusters6"
		pth_file="g36_water6.pth"
		gen_folder="Water6_Gan_structure"
	
		min_std_thresh=1.2
		max_std_thresh=4
		cond_flag=5
		component_NAtom=3



	g_p, d_p,args      = f01_main()
	dataloader,dataset = f02_main(folder,component_NAtom)	
	NAtom              = dataset.configs[0].NAtom

	g_p.NAtom          = NAtom
	d_p.NAtom          = NAtom
#	g_p.Gflag_network  = 2 # =1, use network, =2 not use network

	if component_NAtom>0:
		g_p.NAtom          = int(NAtom/component_NAtom)
		d_p.NAtom          = int(NAtom/component_NAtom)
	


	g_p.configs        = dataset.configs
	discriminator = Discriminator(d_p)	
	generator = Generator(g_p)

	generator,gen_tensor = Gan_train(generator,discriminator,dataset,component_NAtom)
	
	print("pth file save to ", pth_file)
	
	torch.save(generator,pth_file)

	os.makedirs(gen_folder,exist_ok=True)
	min_std_thresh=1
	max_std_thresh=3.5
	for i in range(30):
		filename = f'gan_{i}.xyz'
		filename = os.path.join(gen_folder,filename)
		random_inp   = generator.Gen_Random_input()
		gen_tensor = generator.forward(random_inp) 

		#Gen_XYZ(gen_tensor,dataset.configs[0].atoms,0,filename) #if for clusters
		if component_NAtom==0:
			Gen_XYZ(gen_tensor,dataset.configs[0].atoms,0,min_std_thresh, max_std_thresh,filename)  #if for  org mol
		else:
			Bead_Gen_XYZ(component_NAtom,dataset.configs, gen_tensor,cond_flag, min_std_thresh, max_std_thresh, filename)

	os.system("python3.6 A11_call_generator.py")
