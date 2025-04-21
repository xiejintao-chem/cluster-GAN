import torch
import numpy as np
def ConvertTensor(vec,grad_flag=True):
        return torch.tensor(vec,dtype=torch.float32,requires_grad=grad_flag)

def cal_avg(subset,component_NAtom):
	avg_geom = 0
	for config in subset:
		if component_NAtom==0:
			avg_geom+=config.geom_c
		else:
			avg_geom+=config.bead_center_geom

	Nconfig = len(subset)
	avg_geom =avg_geom/Nconfig
	avg_geom = np.ravel(avg_geom)
	avg_geom = ConvertTensor(avg_geom,True)
	return avg_geom	
