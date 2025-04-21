#lq---
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
from G01_parameters import   ConvertTensor
from A01_util import *
from G05_input_transformation import *

#parse the whole clusters into individual molecules 
#starting_bead_center: the index for the starting bead_center
def Setup_Beads(geom, component_NAtom, starting_bead_center=0):
	dim = geom.shape[0]	
	mols = []
	bead_centers=[]

	natom = component_NAtom
	for i in range(0,dim, natom):
		idx = starting_bead_center + i
		each_mol = geom[ idx : idx+natom, :]
		mols.append(each_mol)
		
		bead_center = geom[idx,:]
		bead_centers.append(bead_center)

	return mols, bead_centers

class Config:
	pass

class myDataset(Dataset):

	def __init__(self,foldername,component_NAtom=0):
		self.configs = self.load_data(foldername,component_NAtom)
		self.component_NAtom =component_NAtom

	def load_data(self,foldername, component_NAtom):
		files  = os.listdir(foldername)
		files.sort()
		configs        = []
		for f in files:
			fn     = os.path.join(foldername,f)
			config = read_xyz(fn,component_NAtom)		

			configs.append(config)

		return configs

	
	def __len__(self):
		return len(self.configs)

	def __getitem__(self,index):
		if self.component_NAtom==0:
			return  ConvertTensor((self.configs[index].geom_c),True)
		else:
			return ConvertTensor((self.configs[index].bead_center_geom),True)         

def read_xyz(filename,component_NAtom):
	data  = pd.read_csv(filename,skiprows=2,header=None,delimiter='\s+')
	geom  = data.iloc[:,1:]
	atoms = data.iloc[:,0]
	NAtom = len(atoms)
	geom  = np.mat(geom)
	
	mean,geom_c  = translate_geom(geom)
	std, geom_sc = scale_geom(geom_c)
	bond_lengths = calc_bond_lengths(geom)
	bond_map     = calc_bond_map(geom)
	upper_tri    = Get_Upper_Triangle(bond_map)
	

	config = Config()
	config.filename = filename
	config.atoms    = list(atoms)
	config.geom     = geom
	config.geom0    = geom.flatten()
	config.NAtom    = NAtom
	config.mean     = mean
	config.std      = std
#	config.iqr      = Interquartile_range(geom_c.flatten())
	config.geom_c   = geom_c.flatten() #translated geom
	config.geom_zscore  = geom_sc.flatten() #translated and scaled geom; Z-score
	config.geom_sig_trans = sig_fun(config.geom_c)

	config.bond_lengths = bond_lengths.flatten()
	config.bond_map = bond_map.flatten()
	config.upper_bond_map = upper_tri.flatten()
	config.len_triu       = len(upper_tri)

	if component_NAtom > 0:
		component_geom_list, bead_centers = Setup_Beads(geom_c, component_NAtom)	
		bead_center_geom  = np.array(bead_centers).flatten()


		config.bead_center_geom = bead_center_geom #has been flattened


	return config

#component_NAtom assigns the Natom for component mol
def f02_main(folder,component_NAtom=0):
	mydataset = myDataset(folder,component_NAtom)
	batch_size = 2
	dataloader=torch.utils.data.DataLoader(mydataset,batch_size=batch_size,shuffle=True,drop_last=True)

	return dataloader,mydataset
if __name__=="__main__":
	case=1
	if case==1:
		folder = "Al12_generated_structures" 
		pth_file = "g36_Al12.pth"
		gen_folder = "Al12_GAN_trained_structures"
		component_NAtom=0
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
		pth_file = "g36_Al4Si7.pth"
		gen_folder = "Al4Si7_Gan_structures"

	if case==7:
		folder="C20_conformers"
		pth_file = "g36_c20.pth"
		gen_folder = "C20_Gan_structures"	

	if case==8:
		folder="h2o_clusters"
		pth_file="g36_water100.pth"
		gen_folder="Water100_Gan_structure"

		component_NAtom=3

	if case==9:
		folder="h2o_clusters6"
		pth_file="g36_water6.pth"
		gen_folder="Water6_Gan_structure"
		component_NAtom=3
		cond_flag=3


	print(folder)
	dataloader,dataset = f02_main(folder,component_NAtom)
#	for idx,geom in enumerate(dataloader):
#		print(idx,geom.shape)
#		print(geom)
#	print(dataset.configs[0].bead_center_list)
#	print(len(dataset.configs[45].bead_center_list))
	for config in dataset.configs:
		print(config.std)
		#print("--")
