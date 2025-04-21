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

from A01_util import *
from G01_parameters import f01_main,ConvertTensor
from G02_dataset import f02_main

class DistancePenalty(nn.Module):
	def __init__(self,info,flag=1):
		super().__init__()
		self.info       = info
		self.thresh_min = info.thresh1  #bond length threshold
		self.thresh_max = info.thresh2
		self.flag       = flag
		self.w1         = 1 #10 occasionally seems best, 20 not as 10 
		self.w2         = 1
		self.w3         = 1
		self.w4         = 1
		

#		torch.manual_seed(16) #2 4,5,11 
		self.std_min    = 1.5
		self.std_max    = 2.0
		self.rand       = self.std_min + ( self.std_max - self.std_min ) *  torch.rand(1)

		self.iqr_min    = 2.0  #si fen wei shu
		self.iqr_max    = 2.9 
		self.iqr_rand   = self.iqr_min + ( self.iqr_max - self.iqr_min ) * torch.rand(1)
		

		self.loss       = nn.MSELoss()

#		print(self.thresh_min,self.thresh_max)
		print("G04 penalty std",self.rand)
		print("G02 iqr",self.iqr_rand) 	
	def forward(self,flatten_geom):

		if self.flag==1:
#			print("penalty flag=1")
			return 0	

		#bond_length penalty
		if self.flag==2:
			pairwise_dist = self.Calc_BondLength(flatten_geom)

			below_thresh_penalty = torch.relu (self.thresh_min - pairwise_dist)
			above_thresh_penalty = torch.relu (pairwise_dist   - self.thresh_max)

			punish_below = below_thresh_penalty.mean()
			punish_above = above_thresh_penalty.mean()
	
	#		punish_below = torch.linalg.norm(below_thresh_penalty) #performs bad
	#		punish_above = torch.linalg.norm(above_thresh_penalty)


			final_punish = self.w1*punish_below + self.w2*punish_above
			return final_punish
		#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
		#attempt1
		#std penalty
		if self.flag==3:
			std  = torch.std(flatten_geom)
			
#		std_below = torch.relu (self.std_min - std)
#		std_above = torch.relu (std - self.std_max)	
#		final_punish = self.w1*punish_below + self.w2*punish_above + self.w3*std_below + self.w4*std_above
#			print(type(std),type(self.rand))
#			print(std.shape,self.rand.shape)

			std_punish = self.loss(std , self.rand)
			return std_punish
		#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

		if self.flag==4: #consider too short bond length and std
			pairwise_dist = self.Calc_BondLength(flatten_geom)
			below_thresh_penalty = torch.relu (self.thresh_min - pairwise_dist)
			punish_below = below_thresh_penalty.mean()

			std  = torch.std(flatten_geom)
			std_punish = self.loss(std , self.rand)

			final_punish = self.w1*punish_below + self.w2*std_punish
			return final_punish
	
		if self.flag==5: #only consider too short bond length; can give good loss, but std too large
			pairwise_dist = self.Calc_BondLength(flatten_geom)
			below_thresh_penalty = torch.relu (self.thresh_min - pairwise_dist)
			punish_below = below_thresh_penalty.mean()

			final_punish = self.w1*punish_below 
			return final_punish
	

		if self.flag==6: #KL div very poor
			#kl_loss = 0
	
			#inp_p    = self.Calc_UpperBondMap(flatten_geom).detach()
#			target_q = ConvertTensor(config.upper_bond_map)
			target_q = ConvertTensor(self.info.configs[0].geom_c)
			kl_loss = self.Calc_KL_div(flatten_geom, target_q,reduction='batchmean')
	
			return kl_loss

		if self.flag==8: #consider too short bond length and std and iqr
			pairwise_dist = self.Calc_BondLength(flatten_geom)
			below_thresh_penalty = torch.relu (self.thresh_min - pairwise_dist)
			punish_below = below_thresh_penalty.mean()

			std  = torch.std(flatten_geom)
			std_punish = self.loss(std , self.rand)

			iqr = self.Calc_quantile(flatten_geom)
			iqr_punish = self.loss(iqr, self.iqr_rand)


			final_punish = self.w1*punish_below + self.w2*std_punish + self.w3*iqr_punish
			return final_punish
	
		#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
		#attempt2 compare the xyz coordiante not good
		#attempt3 compare the bond length
		#calc the kl divergence
		#configs =  self.info.configs
		#for config in configs:	

		##	break

		#final_punish = self.w1*punish_below + self.w2*punish_above + self.w3*kl_loss
		#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


	def Calc_quantile(self,data,percentage1=0.25,percentage2=0.75):
		q1 = torch.quantile(data,percentage1)
		q3 = torch.quantile(data,percentage2)
		iqr = q3 - q1
		return iqr
	
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


	def Calc_UpperBondMap(self,flatten_geom):
		geom = flatten_geom.view(-1,3)
		NAtom = geom.shape[0]
		bond_map = torch.zeros(NAtom,NAtom)
		entries=[]	
		for i in range(NAtom):
			for j in range(i+1,NAtom):
				ai = geom[i,:]
				aj = geom[j,:]
				rij = ai - aj
				rij = torch.linalg.norm(rij)
				entries.append(rij)
		return ConvertTensor(entries)
	



	def Calc_KL_div(self,inp_p,target_q,reduction='batchmean'):
		loss  = f.kl_div(inp_p,target_q,reduction=reduction)

		return loss 	

