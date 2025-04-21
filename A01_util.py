import torch
import numpy as np
import os
import torch.nn as nn
import math
from G05_input_transformation import *
def ConvertTensor(vec,grad_flag=True):
    return torch.tensor(vec,dtype=torch.float32,requires_grad=grad_flag)

def printmat(mat):
        r,c=mat.shape
        for i in range(r):
                for j in range(c):
                        print("%10.6f" %mat[i,j],"   ",end='')
                print()


def Interquartile_range(data):
	q1 = np.percentile(data,25)
	q3 = np.percentile(data,75)
	iqr = q3-q1
	return iqr

def check_grad(tensor):
	print("gradient availability check ",tensor.requires_grad)

#def Polyhedron4(edge_length):  
#	"""生成正四面体的坐标"""  
#	a = edge_length  
#	h = (math.sqrt(6) / 3) * a  
#	geom = np.mat( [  
#        (0, 0, 0),  
#        (a, 0, 0),  
#        (0, a, 0),  
#        (0, 0, h)  
#    ]  )
#
#	return geom,np.std(geom)
#
#def Polyhedron6(edge_length):
#	a = edge_length  
#	geom = np.mat([  
#        (a, a, a), (a, a, -a), (a, -a, a), (a, -a, -a),  
#        (-a, a, a), (-a, a, -a), (-a, -a, a), (-a, -a, -a)  
#    ] )
#
#	return geom, np.std(geom)
#
#def Polyhedron8(edge_length):
#	"""  
#	生成正八面体的顶点坐标。  
#	  
#	:param edge_length: 正八面体棱长。  
#	:return: 一个包含八个顶点坐标的列表。  
#	"""  
#	# 正八面体的一个顶点位于原点  
#	A = (0, 0, 0)  
#	  
#	# 其余顶点位于坐标轴上，距离原点为棱长的一半  
#	half_edge = edge_length / 2  
#	  
#	# x轴正方向和负方向上的顶点  
#	B = (half_edge, 0, 0)  
#	C = (-half_edge, 0, 0)  
#	  
#	# y轴正方向和负方向上的顶点  
#	D = (0, half_edge, 0)  
#	E = (0, -half_edge, 0)  
#	  
#	# z轴正方向和负方向上的顶点  
#	# 由于正八面体的结构，这些点并不是简单地位于z轴上，而是稍微偏离以形成等边三角形面  
#	# 使用勾股定理计算z坐标  
#	z_offset = math.sqrt(edge_length**2 - half_edge**2)  
#	F = (0, 0, z_offset)  
#	G = (0, 0, -z_offset)  
#	  
#	# 返回所有顶点的坐标  
#	geom = np.mat( [A, B, C, D, E, F, G] )
#	return geom,np.std(geom) 
#  
#def Polyhedron12(edge_length):
#	"""  
#	生成正十二面体的顶点坐标。  
#	  
#	:param edge_length: 正十二面体的棱长。  
#	:return: 一个包含20个顶点坐标的列表。  
#	"""  
#	phi = (1 + np.sqrt(5)) / 2
#	
#	
#	# 正十二面体的其中一个顶点，其他顶点通过旋转和平移得到
#	base_vertex = np.array([np.sqrt((2 + phi)/2), np.sqrt((2 - phi)/2), 0])
#	
#	
#	# 正十二面体的旋转矩阵，基于A5群的旋转操作生成其余顶点
#	rotation_matrices = [
#	    # 请参考数学文献或计算得出3组适当的旋转矩阵，这里省略具体实现
#	]
#	
#	
#	# 通过应用不同的旋转矩阵到基础顶点，生成全部顶点
#	vertices = [base_vertex]
#	for rotation in rotation_matrices:
#	    vertices.extend(np.dot(rotation, base_vertex))
#	
#	
#	# 去除重复顶点（由于对称性可能会有重复）
#	vertices = np.unique(vertices, axis=0)
#	
#	
#	# 可选：归一化顶点坐标，确保它们都在单位球面上
#	vertices /= np.linalg.norm(vertices, axis=1)[:, np.newaxis]
# 
#	geom = np.mat(vertices)
#	return geom,np.std(geom)  
#	 
#
#def Polyhedron20(edge_length):
#	"""  
#	生成正二十面体的顶点坐标。  
#	  
#	:param edge_length: 正二十面体的棱长。  
#	:return: 一个包含12个顶点坐标的列表。  
#	"""  
#	phi = (1 + math.sqrt(5)) / 2  # 黄金分割比  
#	radius = edge_length / (2 * math.sqrt(phi**2 + 1))  # 从中心到顶点的半径  
#	  
#	# 计算12个顶点的坐标  
#	# 每个顶点由球坐标系中的(r, theta, phi)转换而来  
#	# 其中r是半径，theta是方位角，phi是仰角  
#	theta = [i * 2 * math.pi / 5 for i in range(5)] + [0]  # 方位角  
#	phi = [math.acos(-1 / phi) if i % 2 == 0 else math.acos(1 / phi) for i in range(5)] + [math.pi / 2]  # 仰角  
#	  
#	vertices = []  
#	for t in theta:  
#		for p in phi:  
#			x = radius * math.sin(p) * math.cos(t)  
#			y = radius * math.sin(p) * math.sin(t)  
#			z = radius * math.cos(p)  
#			vertices.append((x, y, z))  
#	  
#	# 返回所有顶点的坐标  
#	geom = np.mat(vertices)
#	return geom,np.std(geom)
#	
#

def CalcBondLength(geom,i,j):
    ri=geom[i,:]
    rj=geom[j,:]
    rij=ri-rj
    rij=torch.linalg.norm(rij)
    return rij

def np_CalcBondLength(geom,i,j):
    ri=geom[i,:]
    rj=geom[j,:]
    rij=ri-rj
    rij=np.linalg.norm(rij)
    return rij


def CalcAllBondLength(geom):
    NAtom=geom.shape[0]
    res=[]
    bond_lengths={}
    for i in range(NAtom-1):
        for j in range(i+1,NAtom):
            rij = CalcBondLength(geom,i,j)
            res.append([[i,j],rij])
            bond_lengths[str([i,j])]=rij

    return res,bond_lengths

def Calc_CRow_T(NAtom, each_config,thresh=1.6):

	each_geom   = each_config.view(NAtom,3)
	
	CTable      = torch.eye(NAtom)

	for i in range(NAtom-1):
		for j in range(i+1,NAtom):
			rij = CalcBondLength(each_geom,i,j)
			if rij <=thresh:
				CTable[i,j] = 1
				CTable[j,i] = 1	

	CTable.requires_grad_(False)
	CRow = CTable.view(1,NAtom*NAtom)

	return CRow


#def Calc_CRow_T(ref_CRow_T, each_config, thresh=1.6): #gen_geom should be batch_size,threeN
#	NAtom       = int(len(each_config) / 3 )
#	each_geom_T = each_config.view(NAtom,3)
#
#	bondlengthlist,bond_length_dict=CalcAllBondLength(each_geom_T)
#	ConnectTable = torch.zeros(NAtom,NAtom)
#	ConnectTable.requires_grad_(True)
#	
#	for i in range(NAtom):
#		for j in range(NAtom):
#			key = str([i,j])
#			if key in bond_length_dict.keys():
#				value = bond_length_dict[key]


def Tensor_Connectivity(NAtom,flat_geom_T,batch_size, ref_CTable_T, thresh=1.6):

	def calc_one_ctable(flat_geom_t):
		geom_T = flat_geom_t.view(NAtom,3)
	 
		bondlengthlist,bond_length_dict=CalcAllBondLength(geom_T)
	
		ConnectTable = np.mat(np.zeros((NAtom,NAtom)),dtype=int)
		
		for i in range(NAtom):
		    for j in range(NAtom):
		        key = str([i,j])
		        if key in bond_length_dict.keys():
		            value = bond_length_dict[key]
		        else:
		            continue
		
		        if value<=thresh:
		            ConnectTable[i,j]=1
		ConnectTable = np.reshape(ConnectTable,(1,NAtom*NAtom))
		gen_CRow_T   = torch.tensor(ConnectTable,dtype=torch.float32,requires_grad=True)

		return ConnectTable,gen_CRow_T

	if batch_size==1:
		ConnectTable,gen_CRow_T = calc_one_ctable()
		diff = ref_CTable_T - gen_CRow_T
		diff = torch.linalg.norm(diff)

	else:
		batch_size = flat_geom_T.shape[0]

		diff = 0		
		for i in range(batch_size):
			flat_geom = flat_geom_T[i,:].detach()
			ConnectTable,gen_CRow_T = calc_one_ctable(flat_geom)
			delta = ref_CTable_T - gen_CRow_T
			delta = torch.linalg.norm(delta)
			diff += delta

	return diff
#calculate the space between each mol in the cluster
def mol_space(geom,component_NAtom,thresh=1):
	NAtom = geom.shape[0]
	dim = int(NAtom/component_NAtom)

	bond_length=[]
	for i in range(dim):
		for j in range(i+1,dim):
			idx1 = i*component_NAtom
			idx2 = j*component_NAtom
			r00=np_CalcBondLength(geom,idx1+0,idx2+0)	
			r01=np_CalcBondLength(geom,idx1+0,idx2+1)	
			r02=np_CalcBondLength(geom,idx1+0,idx2+2)	
			r10=np_CalcBondLength(geom,idx1+1,idx2+0)	
			r11=np_CalcBondLength(geom,idx1+1,idx2+1)	
			r12=np_CalcBondLength(geom,idx1+1,idx2+2)	
			r20=np_CalcBondLength(geom,idx1+2,idx2+0)	
			r21=np_CalcBondLength(geom,idx1+2,idx2+1)	
			r22=np_CalcBondLength(geom,idx1+2,idx2+2)	

			bond_length.append(r00)
			bond_length.append(r01)
			bond_length.append(r02)
			bond_length.append(r10)
			bond_length.append(r11)
			bond_length.append(r12)
			bond_length.append(r20)
			bond_length.append(r21)
			bond_length.append(r22)
	
	return bond_length

def Bead_Gen_XYZ(component_NAtom,configs, out,cond_flag=1,min_std_thresh=1.2, max_std_thresh=2, filename="LQ_train_res.xyz"):
	geom = out.detach().numpy()
	mean = np.mean(geom)
	std  = np.std(geom)

	geom = np.reshape(geom, (-1,3))

	#
	rand_idx = np.random.randint(len(configs))
	ref_geom = configs[rand_idx].geom_c
	ref_geom = np.reshape(ref_geom,(-1,3))


	nmol = int(configs[0].NAtom/component_NAtom)
#	print(type(geom))
#	print(type(ref_geom))

	for i in range(nmol):
		idx = i*component_NAtom
		mol_geom = ref_geom[idx:idx+3,:]
		bead_geom = geom[i,:]
		
		ref_geom[idx:idx+3,:] = mol_geom - ref_geom[idx,:] + bead_geom 
	
	geom = ref_geom
	#---------------------------------
	if cond_flag==1:
		#condition 1
		#check bond length
		min_thresh=1.7
		bond_length = calc_bond_lengths(geom)
		min_bl = min(bond_length)
		if min_bl < min_thresh:	
			#print(filename,"too short bond length,pass")	
			return 0
	
	if cond_flag==2:
		max_thresh=8
		#condition 2
		max_bl = max(bond_length)
		if max_bl > max_thresh:
			#print(filename,"too long bond length,pass")
			return 0
	
	if cond_flag==3:
		#condition 3
		#check std
		min_std_thresh = min_std_thresh #1.5
		max_std_thresh = max_std_thresh  #2.2
		if std < min_std_thresh or std > max_std_thresh:
			#print(filename,"std not fit")
			return 0
	
		#check bond length
		min_thresh=0.9 #hardcode constant
		bond_length = calc_bond_lengths(geom)
		min_bl = min(bond_length)
		if min_bl < min_thresh:	
			#print(filename,"too short bond length,pass")	
			return 0


		#every H/F should has one neighboring C	
		#the first 10 atoms are C
		NAtom_C = 10 #hardcode constant
		NAtom_H = geom.shape[0]-NAtom_C
#		C_atom_geom  = geom[0:10]
		H_atom_geom  = geom[NAtom_C:]

		for i in range(NAtom_C):
			C_geom = geom[i,:]
			CH=[]
			for j in range(NAtom_H):
				H_geom = H_atom_geom[j,:]
				dist = H_geom-C_geom
				dist = np.linalg.norm(dist)
				CH.append(dist)

			min_CH = min(CH)
			minCH_thresh = 1.3 #hardcode constant
			if min_CH > minCH_thresh:
				return 0	
				
	if cond_flag==4:
		#condition 4
		#check std
		min_std_thresh = min_std_thresh #1.5
		max_std_thresh = max_std_thresh  #2.2
		if std < min_std_thresh or std > max_std_thresh:
			#print(filename,"std not fit")
			return 0
	

	if cond_flag==5:
		#condition 5
		#check std
		min_std_thresh = min_std_thresh #1.5
		max_std_thresh = max_std_thresh  #2.2
		if std < min_std_thresh or std > max_std_thresh:
			#print(filename,"std not fit")
			return 0
	
		#check bond length
		min_thresh=0.9 #hardcode constant
		bond_length = calc_bond_lengths(geom)
		min_bl = min(bond_length)
		if min_bl < min_thresh:	
			#print(filename,"too short bond length,pass")	
			return 0

		#check spacing between mols
		mol_space_bond_length = mol_space(geom,component_NAtom)
		mol_space_thresh=1.5 #hardcode constant
		min_bl2 = min(mol_space_bond_length)

		if min_bl2 < mol_space_thresh:	
			#print(filename,"too short molecular inter-distance,pass")	
			return 0

	
	


	#---------------------------------

	print("mean of generated geom: ", np.round(mean,3))
	print("std of generated geom:  ", np.round(std,3))



	#write to xyzfile
	outfile = "tmp.xyz"
	np.savetxt(outfile,geom,fmt="%.6f")


	NAtom = configs[0].NAtom
	atoms = configs[0].atoms
	with open(outfile,"r") as fn:
		lines=fn.readlines()
	new_line = f"{NAtom}\n\n"
	lines.insert(0,new_line)
	with open(outfile,"w") as fn:
		fn.writelines(lines)

	print("generated xyz saved in ", filename)
	f=open(outfile)
	fw=open(filename,"w")
	lines = f.readlines()
	fw.write(lines[0])
	fw.write(lines[1])

	lines=lines[2:]
	count=0
	for line in lines:
		atom = atoms[count]
		new_line = atom+" " +line
		fw.write(new_line)
		count+=1
	f.close()
	fw.close()
	os.remove(outfile)
	return 1


#read in flatten geom
def Gen_XYZ(out,atoms,cond_flag=1,min_std_thresh=1.2, max_std_thresh=2, filename="LQ_train_res.xyz"):
	geom = out.detach()
	mean = np.mean(geom.numpy())
	std  = np.std(geom.numpy())

	NAtom = len(atoms)
	geom = np.reshape(geom,(NAtom,3))

	
	#---------------------------------
	if cond_flag==1:
		#condition 1
		#check bond length
		min_thresh=1.7
		bond_length = calc_bond_lengths(geom)
		min_bl = min(bond_length)
		if min_bl < min_thresh:	
			#print(filename,"too short bond length,pass")	
			return 0
	
	if cond_flag==2:
		max_thresh=8
		#condition 2
		max_bl = max(bond_length)
		if max_bl > max_thresh:
			#print(filename,"too long bond length,pass")
			return 0
	
	if cond_flag==3:
		#condition 3
		#check std
		min_std_thresh = min_std_thresh #1.5
		max_std_thresh = max_std_thresh  #2.2
		if std < min_std_thresh or std > max_std_thresh:
			#print(filename,"std not fit")
			return 0
	
		#check bond length
		min_thresh=0.9 #hardcode constant
		bond_length = calc_bond_lengths(geom)
		min_bl = min(bond_length)
		if min_bl < min_thresh:	
			#print(filename,"too short bond length,pass")	
			return 0


		#every H/F should has one neighboring C	
		#the first 10 atoms are C
		NAtom_C = 10 #hardcode constant
		NAtom_H = geom.shape[0]-NAtom_C
#		C_atom_geom  = geom[0:10]
		H_atom_geom  = geom[NAtom_C:]

		for i in range(NAtom_C):
			C_geom = geom[i,:]
			CH=[]
			for j in range(NAtom_H):
				H_geom = H_atom_geom[j,:]
				dist = H_geom-C_geom
				dist = np.linalg.norm(dist)
				CH.append(dist)

			min_CH = min(CH)
			minCH_thresh = 1.3 #hardcode constant
			if min_CH > minCH_thresh:
				return 0	
				
	if cond_flag==4:
		#condition 3
		#check std
		min_std_thresh = min_std_thresh #1.5
		max_std_thresh = max_std_thresh  #2.2
		if std < min_std_thresh or std > max_std_thresh:
			#print(filename,"std not fit")
			return 0
	
		#check bond length
		min_thresh=0.9 #hardcode constant
		bond_length = calc_bond_lengths(geom)
		min_bl = min(bond_length)
		if min_bl < min_thresh:	
			#print(filename,"too short bond length,pass")	
			return 0

	#---------------------------------

	print("mean of generated geom: ", np.round(mean,3))
	print("std of generated geom:  ", np.round(std,3))



	#write to xyzfile
	outfile = "tmp.xyz"
	np.savetxt(outfile,geom,fmt="%.6f")

#	os.system("sed -i '1i%d' %s" %(NAtom,outfile))
#	os.system("sed -i '1a\n' %s" %(outfile))


	with open(outfile,"r") as fn:
		lines=fn.readlines()
	new_line = f"{NAtom}\n\n"
	lines.insert(0,new_line)
	with open(outfile,"w") as fn:
		fn.writelines(lines)

	print("generated xyz saved in ", filename)
	f=open(outfile)
	fw=open(filename,"w")
	lines = f.readlines()
	fw.write(lines[0])
	fw.write(lines[1])

	lines=lines[2:]
	count=0
	for line in lines:
		atom = atoms[count]
		new_line = atom+" " +line
		fw.write(new_line)
		count+=1
	f.close()
	fw.close()
	os.remove(outfile)
	return 1

def calc_bond_map(geom):
    NAtom = geom.shape[0]
    bond_map = np.zeros((NAtom,NAtom))

    for i in range(NAtom):
        for j in range(NAtom):
            ai = geom[i,:]
            aj = geom[j,:]
            rij = ai - aj
            rij = np.linalg.norm(rij)
            bond_map[i,j]=rij


    return bond_map

def Get_Upper_Triangle(mat):
    r,c = mat.shape
    elements=[]
    for i in range(r):
        for j in range(i+1,c):
            elements.append(mat[i,j])
    return np.array(elements)

def translate_geom(geom):
    mean = np.mean(geom)
    new_geom = geom-mean
    return mean,new_geom

def scale_geom(geom):
    std = np.std(geom)
    new_geom = geom/std
    return std,new_geom

def calc_bond_lengths(geom):
    flag=0
    if isinstance(geom,torch.Tensor):
#       geom=geom.detach().numpy()
        flag=1

    Natom = geom.shape[0]
    bond_lengths=[]
    for i in range(Natom):
        for j in range(i+1,Natom):
            ai = geom[i,:]
            aj = geom[j,:]
            rij = ai - aj
            if flag==0:
                rij = np.linalg.norm(rij)
            else:
                rij = torch.linalg.norm(rij)
            bond_lengths.append(rij)

    bond_lengths.sort()
    if flag==0:
        bond_lengths = np.array(bond_lengths)
    else:
        bond_lengths = ConvertTensor(bond_lengths)

    return bond_lengths


#------------------------
class Linear(nn.Linear):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        bias: bool = True,
        init: str = "default",
    ):
        super(Linear, self).__init__(d_in, d_out, bias=bias)

        self.use_bias = bias

        if self.use_bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init == "default":
            self._trunc_normal_init(1.0)
        elif init == "relu":
            self._trunc_normal_init(2.0)
        elif init == "glorot":
            self._glorot_uniform_init()
        elif init == "gating":
            self._zero_init(self.use_bias)
        elif init == "normal":
            self._normal_init()
        elif init == "final":
            self._zero_init(False)
        elif init == "jax":
            self._jax_init()
        else:
            raise ValueError("Invalid init method.")

    def _trunc_normal_init(self, scale=1.0):
        # Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        TRUNCATED_NORMAL_STDDEV_FACTOR = 0.87962566103423978
        _, fan_in = self.weight.shape
        scale = scale / max(1, fan_in)
        std = (scale**0.5) / TRUNCATED_NORMAL_STDDEV_FACTOR
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std)

    def _glorot_uniform_init(self):
        nn.init.xavier_uniform_(self.weight, gain=1)

    def _zero_init(self, use_bias=True):
        with torch.no_grad():
            self.weight.fill_(0.0)
            if use_bias:
                with torch.no_grad():
                    self.bias.fill_(1.0)

    def _normal_init(self):
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity="linear")

    def _jax_init(self):
        input_size = self.weight.shape[-1]
        std = math.sqrt(1 / input_size)
        nn.init.trunc_normal_(self.weight, std=std, a=-2.0 * std, b=2.0 * std)

class MLP(nn.Module):
    def __init__(self,d_in,n_layers,d_hidden,d_out,activation=nn.ReLU(),bias=True):
        super(MLP, self).__init__()
        layers = [Linear(d_in, d_hidden, bias), activation]
        for _ in range(n_layers):
            layers += [Linear(d_hidden, d_hidden, bias), activation]
        layers.append(Linear(d_hidden, d_out, bias))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

