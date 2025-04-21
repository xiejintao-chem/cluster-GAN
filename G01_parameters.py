import torch
from torch.autograd import Variable
import os
import numpy as np
import argparse
import torch.nn as nn
import copy
import random
import pandas as pd
def ConvertTensor(vec,grad_flag=True):
	return torch.tensor(vec,dtype=torch.float32,requires_grad=grad_flag)


class G_Parameter():
	def __init__(self):
		activation_funcs=[nn.ReLU(), #0    [0,inf]
			              nn.Tanh(), #1    [-1,1]
			              nn.Sigmoid(),#2  [0,1)
			              nn.Softmax(),#3  [0,1)
			              nn.ReLU6(),#4
			              nn.LeakyReLU(),#5 [-inf,inf]
						  nn.LeakyReLU(0.5) #6
	                      ]
		loss_funcs=[nn.MSELoss(),             nn.BCELoss(),           nn.L1Loss(),
			    nn.NLLLoss(),             nn.PoissonNLLLoss(),    nn.KLDivLoss(),
                            nn.BCEWithLogitsLoss(),   nn.MarginRankingLoss(),
                            nn.HingeEmbeddingLoss(),  nn.MultiLabelMarginLoss(),
                            nn.SmoothL1Loss(),        nn.SoftMarginLoss(),
                            nn.CosineEmbeddingLoss(), nn.MultiLabelSoftMarginLoss(),
                            nn.MultiMarginLoss(),     nn.TripletMarginLoss(),
                            nn.CTCLoss()]
	
		optimizers=[torch.optim.Adam,
                    torch.optim.SGD,
                    torch.optim.NAdam,
                    torch.optim.RAdam,
                    torch.optim.LBFGS,
		    torch.optim.RMSprop
                      ]
		Normalization_method=[None,nn.BatchNorm1d,nn.LayerNorm]


		args  = ParseInp()
		self.args      = args 


		self.activation_func = activation_funcs[ self.args.G_activation_func ]
		self.loss_func       = loss_funcs[ self.args.G_loss_func ]
		self.optimizer       = optimizers[ self.args.G_optimizer ]  #choose the optimizer
		#model1 simple;
		#model2 increase decrease; 
		#model3 decrease increase; 
		#model4 model 2 but not use activation
		#model5 same number of neurons (latent_dim)
		#model6 MLP
		#model7 ResNet
		self.latent_dim      = self.args.G_latent_dim #186  #only for model 5
		self.increment       = self.args.G_increment
		self.Nlayer          = self.args.G_Nlayer
		self.model_flag      = self.args.G_model_flag
		self.batch_size      = self.args.G_batch_size
		self.lr				 = self.args.G_lr
		self.Normalization   = Normalization_method[self.args.G_norm_method]
		self.maxit           = self.args.G_maxit
		self.penalty_flag    = self.args.G_penalty_flag
		if self.Normalization == None:
			pass
		else: #the negative flag has normalization
			self.model_flag  = self.model_flag+100

		#---------------------------------#
		#miscellaneous options
		self.thresh1         = self.args.M_thresh1
		self.thresh2         = self.args.M_thresh2
		self.avg_bond_length = self.args.M_avg_bond_length
		self.Epoch           = self.args.M_Epoch
	#generator
	def __str__(self):
		print("Object of class G_Parameter:\n")
		print("activation_function = ", self.activation_func)	
		print("loss_function       = ", self.loss_func)
		print("optimizer           = ", self.optimizer)
		print("learning rate       = ", self.lr,"\n")
		print("Normalizer          = ", self.Normalization)
		print("model flag          = ", self.model_flag)
		print("increment           = ", self.increment)
		print("Nlayer              = ", self.Nlayer,"\n")
	#	print("latent_dim          = ", self.latent_dim,"\n")
		print("G maxit             = ", self.maxit)
		print("G penalty flag      = ", self.penalty_flag)
	#	print("ref geom file       = ", self.ref_XYZ_file)
	#	print("ref CTable file     = ", self.ref_CTable_file,"\n")
	#	print("NAtom               = ", self.NAtom)
	#	print("bond thresh 4 CTable (Ang) = ", self.thresh)
			

		return "-----------------------------------------------------\n"
class D_Parameter():
	def __init__(self):

		activation_funcs=[nn.ReLU(),#0
			              nn.Tanh(),#1
			              nn.Sigmoid(),#2
			              nn.Softmax(),#3
			              nn.ReLU6(),#4
			              nn.LeakyReLU(),#5
						  nn.LeakyReLU(0.2)#6
	                      ]
		loss_funcs=[nn.MSELoss(),nn.BCELoss()]
	
		optimizers=[torch.optim.Adam,
                    torch.optim.SGD,
                    torch.optim.NAdam,
                    torch.optim.RAdam,
                    torch.optim.LBFGS,
                    torch.optim.RMSprop
                      ]

		Normalization_method=[None,nn.BatchNorm1d,nn.LayerNorm]

		args = ParseInp()
		self.args      = args


		self.activation_func  = activation_funcs[self.args.D_activation_func]
		self.loss_func        = loss_funcs[self.args.D_loss_func]
		self.optimizer        = optimizers[self.args.D_optimizer]  #choose the optimizer
		self.lr               = self.args.D_lr
		self.model_flag       = self.args.D_model_flag
		self.Nlayer           = self.args.D_Nlayer
		self.increment        = self.args.D_increment
	
		self.batch_size       = self.args.D_batch_size

		self.Normalization    = Normalization_method[self.args.D_norm_method]

		if self.Normalization == None:
			pass
		else: #the negative flag has normalization
			self.model_flag   = self.model_flag+100


	def __str__(self):
		print("Object of class D_Parameter:\n")
		print("activation_function = ", self.activation_func)	
		print("loss_function       = ", self.loss_func)
		print("optimizer           = ", self.optimizer)
		print("learning rate       = ", self.lr,"\n")
		print("Normalizer          = ", self.Normalization)
		print("model flag          = ", self.model_flag)
		print("increment           = ", self.increment)
		print("Nlayer              = ", self.Nlayer,"\n")
#		print("batch size          = ", self.batch_size)
		print("Normlization method = ", self.Normalization)
		return "-----------------------------------------------------\n"

def ParseInp():

#	 parser.add_argument('command',help="'train' or 'evaluate'")

    # Parse command line arguments
	parser = argparse.ArgumentParser(description='myGAN')

	parser.add_argument('--G_activation_func', type=int,default=1)
	parser.add_argument('--G_loss_func', type=int,default=0)
	parser.add_argument('--G_optimizer', type=int, default=5)
	parser.add_argument('--G_model_flag', type=int,default=2) #numbers over 100  has normalizer
	parser.add_argument('--G_latent_dim', type=int, default=0) #here latent_dim=ThreeN
	parser.add_argument('--G_Nlayer', type=int,  default=3)
	parser.add_argument('--G_increment', type=int, default=0)
	parser.add_argument('--G_batch_size', type=int, default=2)
	parser.add_argument('--G_lr', type=float, default=0.001)
	parser.add_argument('--G_norm_method',type=int,default=2) #0 not use normalization, 1 BatchNorm1d,2 LayerNorm
	parser.add_argument('--G_maxit',type=int,default=50) 
	parser.add_argument('--G_penalty_flag',type=int,default=1) #1 not use penalty 



	parser.add_argument('--D_activation_func', type=int,default=2) # for gdb 2 4 5
	parser.add_argument('--D_loss_func', type=int,default=0)
	parser.add_argument('--D_optimizer', type=int, default=5)
	parser.add_argument('--D_model_flag', type=int,default=2)
	parser.add_argument('--D_latent_dim', type=int, default=186)
	parser.add_argument('--D_Nlayer', type=int,  default=2)
	parser.add_argument('--D_increment', type=int, default=-6)
	parser.add_argument('--D_batch_size', type=int, default=2)
	parser.add_argument('--D_lr', type=float, default=0.001)
	parser.add_argument('--D_norm_method',type=int,default=2) #0 not use normalization, 1 BatchNorm1d,2 LayerNorm

	parser.add_argument('--M_thresh1', type=float, default=1.6)  # to detect if atoms are too close
	parser.add_argument('--M_thresh2', type=float, default=12.6) #to detect if atoms are too far
	parser.add_argument('--M_avg_bond_length',type=float,default=2.1)#estimated average bond length
	parser.add_argument('--M_Epoch', type=int,default=100)
	
	args= parser.parse_args()


	return args

def f01_main():
	args = ParseInp()
	g_p = G_Parameter()
	d_p = D_Parameter()
	print(g_p)	
	print(d_p)
	return g_p, d_p,args
if __name__=="__main__":

	g_p, d_p,args = f01_main()

