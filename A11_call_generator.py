import torch
import pandas as pd
from G16_generator import *
from A02_aux import *
def make_orca_input(filename,spin=1):
	data  = pd.read_csv(filename,skiprows=2,header=None,delimiter='\s+')
	geom  = data.iloc[:,1:]
	atoms = data.iloc[:,0]
	geom  = np.mat(geom)

	basename = filename.split(".")[0]
	inpfile  = basename+".inp"
	fw = open(inpfile,"w")
#	fw.write("!pal TPSSh pcseg-1 tightSCF grid4 opt numfreq def2/J  nopop\n")
	fw.write("!pal blyp def2-tzvp tightSCF grid4 opt freq  nopop\n")

	fw.write("!angs\n")
	fw.write("%pal nproc 4 end\n")
	fw.write("%MaxCore 4000 \n\n")
	fw.write("%geom\n")
	fw.write("maxiter 200\n")
	fw.write("end\n")

	fw.write("* xyz 0 %i\n" %spin)
	NAtom = len(atoms)
	for i in range(NAtom):
		line = f'{atoms[i]}  {geom[i,0]}  {geom[i,1]}  {geom[i,2]}\n'
		fw.write(line)
	fw.write("*")
	fw.close()

def check_unique_geom(container,gen_tensor, thresh=1):
	geom = gen_tensor.detach().numpy()
	if len(container)==0:
		container.append(geom)
		return 1
	else:
		for g in container:
			diff = g-geom
			diff = np.linalg.norm(diff)
			if diff < thresh:
				return 0

	return 1
	
if __name__=="__main__":

	case=1
	Ngen="100"
	component_NAtom=0
	if case==1:
		folder = "Al12_generated_structures" 
		pth_file = "g36_Al12.pth"
		gen_folder = "Al12_GAN_trained_structures_"+Ngen
		spin = 1
		min_std_thresh=1.5
		max_std_thresh=2.2
		cond_flag=1


	if case==2:
		folder = "P11p_generated_structures"
		pth_file = "g36_P11.pth"
		gen_folder = "P11_GAN_structures_"+Ngen
		spin = 4
		min_std_thresh=1.5
		max_std_thresh=2.2
		cond_flag=1

	if case==3:
		folder="Mg10_generated_structures"
		pth_file = "g36_Mg10.pth"
		gen_folder = "Mg10_Gan_structures_"+Ngen
		spin = 1
		min_std_thresh=1.5
		max_std_thresh=2.2
		cond_flag=1


	if case==4:
		folder="Na10_generated_structures"
		pth_file = "g36_Na10.pth"
		gen_folder = "Na10_Gan_structures_"+Ngen
		spin = 1
		min_std_thresh=1.5
		max_std_thresh=2.2
		cond_flag=1


	if case==5:
		folder="Al4Si7_generated_structures"
		pth_file = "g36_Al4Si7_att4.pth"
		gen_folder = "Al4Si7_Gan_structures_att4_3_"+Ngen
		spin = 3
		min_std_thresh=1.5
		max_std_thresh=2.2
		cond_flag=1


	if case==6:
		print(case)
		folder = "subset108_4_GAN"
		pth_file = "g36_Org108.pth"
		gen_folder = "Org108_Gan_structures"
		spin=1
		min_std_thresh=1
		max_std_thresh=3.5

		cond_flag=3

	if case==7:
		folder="C20_conformers"
		pth_file = "g36_c20.pth"
		gen_folder = "C20_Gan_structures"	
		spin=1

		min_std_thresh=3
		max_std_thresh=5
		cond_flag=4

	if case==8:
		folder="h2o_clusters"
		pth_file="g36_water100.pth"
		gen_folder="Water100_Gan_structure"
		spin=1

		min_std_thresh=2
		max_std_thresh=3
		cond_flag=4
		component_NAtom=3


	if case==9:
		folder="h2o_clusters6"
		pth_file="g36_water6.pth"
		gen_folder="Water6_Gan_structure"
		spin=1


		min_std_thresh=1.2
		max_std_thresh=4
		cond_flag=5
		component_NAtom=3

	print("generating gan generated structures")
	print("using pth file ", pth_file)
	
	folder_basename = folder.split("_")[0]
	
	g_p, d_p,args      = f01_main()
	dataloader,dataset = f02_main(folder)	
	NAtom              = dataset.configs[0].NAtom
	g_p.NAtom          = NAtom
	d_p.NAtom          = NAtom
	g_p.configs        = dataset.configs
	
	
	generator = torch.load(pth_file)
	print(generator.model)
#	gen_folder = folder_basename+"_GAN_trained_structures"
	print("generated structrure saved to ", gen_folder)
	os.makedirs(gen_folder,exist_ok=True)
	hit = 1
	count=1
	container=[]

	print("condition flag",cond_flag)
	while(1):
		filename = f'gan_{str(hit).zfill(2)}.xyz'
		filename = os.path.join(gen_folder,filename)
		#method1
		random_inp   = generator.Gen_Random_input()
	
		#method2
	#	idx = random.randint(0,len(dataset.configs)-1) #choose a random number to setup subset
	#	random_inp = np.ravel(dataset.configs[idx].geom_c)
	#	random_inp = ConvertTensor(random_inp)
	
		#method3
	#	Nconfig = random.randint(43,len(dataset.configs)) #choose a random number to setup subset
	#	subset = random.sample(dataset.configs,Nconfig)
	#	avg_geom=cal_avg(subset)
	#	random_inp = ConvertTensor(avg_geom)
	
		gen_tensor = generator.forward(random_inp) 

#		res1 = check_unique_geom(container,gen_tensor)
#		if res1==0:
#			print("identical geom")
#			container.append(gen_tensor.detach().numpy())
#			print(len(container))

#		res = Gen_XYZ(gen_tensor,dataset.configs[0].atoms,1, filename) #if for cluster; hit+=1 if res=1  
		if component_NAtom==0: 
			res = Gen_XYZ(gen_tensor,dataset.configs[0].atoms,cond_flag, min_std_thresh, max_std_thresh, filename) #if for org mol. hit+=1 if res=1 
		else:
			res = Bead_Gen_XYZ(component_NAtom,dataset.configs, gen_tensor,cond_flag, min_std_thresh, max_std_thresh, filename) #if for org mol. hit+=1 if res=1 



		if res==1:

			res2 = check_unique_geom(container,gen_tensor,thresh=1)
			if res2==0:
				print("generate non-unique structure")
				continue

			print(filename)
			make_orca_input(filename,spin)
			count=1
			print(len(container))
			container.append(gen_tensor.detach().numpy())
		hit+=res
		if hit==101:
			break
		
		count+=1
		if count%10000==0:
			print(count)
	
	
	
