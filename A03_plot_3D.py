import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pandas as pd

import open3d as o3d
def read_xyz(filename):
	data  = pd.read_csv(filename,skiprows=2,header=None,delimiter='\s+')
	geom  = data.iloc[:,1:]
	atoms = data.iloc[:,0]
	NAtom = len(atoms)
	geom  = np.mat(geom)
	

	return geom


def plot3D(geom_list1,geom_list2):
	fig=plt.figure()
	ax = fig.add_subplot(111,projection='3d')
	for geom in geom_list1:
		x=geom[:,0]
		y=geom[:,1]
		z=geom[:,2]
		x=np.ravel(x)
		y=np.ravel(y)
		z=np.ravel(z)
		ax.plot(x,y,z,color='blue')

	for geom in geom_list2:
		x=geom[:,0]
		y=geom[:,1]
		z=geom[:,2]
		x=np.ravel(x)
		y=np.ravel(y)
		z=np.ravel(z)
		ax.plot(x,y,z,color='orange')
	ax.legend()
	plt.show()



def open3D(geom_list1,geom_list2):

	#test = np.random.rand(100, 3)  
	red = np.array([255,0,0])
	gray = np.array([65,65,65]) #[105,105,105]])
	blue = np.array([0,0,255])
	orange = np.array([255,140,0])
	white =np.array([255,255,255])
	black=np.array([0,0,0])
	#setup pcd for geom_list1	
	NAtom = geom_list1[0].shape[0]
	colors1=[]
	for i in range(NAtom):
		if i%3==0:
			_color = red + np.array([-i*1,i*1,i*1])
			print(_color)
			colors1.append(_color)
		else:
			colors1.append(white)
	colors1=np.array(colors1)

	pcd_list = []
	for geom in geom_list1:
		pcd= o3d.geometry.PointCloud()
		geom = np.array(geom)
		pcd.points=o3d.utility.Vector3dVector(geom)
		pcd.colors = o3d.utility.Vector3dVector(colors1)
		pcd_list.append(pcd)

	#setup pcd for geom_list2
	NAtom2 = geom_list2[0].shape[0]
	colors2=[]
	for i in range(NAtom2):
		if i%3==0:
			colors2.append(blue)
		else:
			colors2.append(white)
	colors2=np.array(colors2)

	pcd_list2 = []
	for geom in geom_list2:
		pcd= o3d.geometry.PointCloud()
		geom = np.array(geom)
		pcd.points=o3d.utility.Vector3dVector(geom)
		pcd.colors = o3d.utility.Vector3dVector(colors2)
		pcd_list2.append(pcd)


	pcd_list.extend(pcd_list2)

	coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0, origin=[0, 0, 0])
#	pcd_list.extend(coordinate_frame)
	pcd_list.append(coordinate_frame)


#	# 使用VisualizerWithEditing创建可视化窗口
#	vis = o3d.visualization.Visualizer()
#	vis.create_window()
#	
#	# 添加几何图形到可视化窗口
#	vis.add_geometry(pcd_list)
#	
#	# 获取渲染选项并设置材质属性
#	render_option = vis.get_render_option()
#	render_option.point_size = 5.0  # 设置点大小
#	
#	# 设置透明度需要通过Material来完成
#	material = o3d.visualization.rendering.MaterialRecord()
#	material.shader = 'defaultUnlit'  # 对于透明效果，通常不需要光照计算
#	material.base_color = [0, 0, 1, 0.5]  # RGBA, 最后一个值是透明度，范围是0-1
##	material.point_size = 5.0  # 可选：在这里也可以设置点大小
#	
#	# 更新点云材质
#	vis.update_geometry(pcd_list)
#	vis.capture_screen_image("transparent_pointcloud.png")  # 捕获屏幕图像以查看效果
	


	o3d.visualization.draw_geometries(pcd_list)

def loadXYZ(folder):
	files=os.listdir(folder)
	files.sort()

	geom_list = []
	for f in files:
		if f.endswith(".xyz"):
			f=os.path.join(folder, f)
			geom = read_xyz(f)
			geom_list.append(geom)
	return geom_list

if __name__=="__main__":

	folder1="h2o_clusters6"
	folder2="Water6_Gan_structure"

	database_geom_list = loadXYZ(folder1)
	gan_geom_list = loadXYZ(folder2)
#	plot3D(database_geom_list[:10], gan_geom_list[:10])
	open3D(database_geom_list[:10], gan_geom_list[:10])
	
