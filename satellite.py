from skimage.morphology import medial_axis, skeletonize
from skimage import morphology, filters
import cv2
import math
from skan import skeleton_to_csgraph
from skimage import io, morphology
from skimage import io, morphology
import matplotlib.pyplot as plt
import skimage.measure
import numpy as np
import numpy as np
import cv2
import matplotlib.pyplot as plt

class road_map:
	def __init__(self,image):
		self.image=image
	def make_skeleton(self):
		
		lower_white = np.array([254, 254, 254 ])
		upper_white = np.array([255, 255, 255])
		white_range = cv2.inRange(self.image, lower_white,upper_white)
		self.image[white_range!=0] = [255,255,255]
		lower_yellow = np.array([160, 230, 250 ])
		upper_yellow = np.array([178, 249, 260])
		yellow_range = cv2.inRange(self.image, lower_yellow,upper_yellow)
		self.image[yellow_range!=0] = [255,255,255]
		self.image[np.where((im!=[255,255,255]).all(axis=2))] = [0,0,0]

		self.image= cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		self.image[np.where((self.image!=[255]))] = [0]
		def nearest_nonzero_idx_v2(self,a,x,y):
		a.astype(int)
		nonzero = cv2.findNonZero(a)
		
		distances = np.sqrt((nonzero[:,:,0] - x) ** 2 + (nonzero[:,:,1] - y) ** 2)
		nearest_index = np.argmin(distances)
		
					  
		return nonzero[nearest_index]


	def skeletonization(self):




		roadwidth=[]
		
		
		self._img = np.uint8(self._img)
		binary = self._img > filters.threshold_otsu(self._img)
		
		np.unique(binary)
		skel, distance = medial_axis(binary, return_distance=True)
		dist_on_skel = distance * skel
	
		cv2.imwrite("skelssat"+str(self._tel)+".png",heatmap)
		#find crossings 
		_, _, degrees = skeleton_to_csgraph(skel)
		intersection_matrix = degrees > 2
		
		#list with crossings
		lists=np.argwhere(intersection_matrix== True)
		#set crossingpixel to zero
		logger=dist_on_skel.copy()
		
		for teller,k in enumerate(dist_on_skel):
			for teller2,l in enumerate(k):
				if int(l)!=0:
		
					dist_on_skel[teller][teller2]=255
		for k in lists:
			dist_on_skel[k[0]][k[1]]=0
		retval, labels = cv2.connectedComponents(np.uint8(dist_on_skel))
		list_of_labels=[]
		list_of_lanes=[]
		indexes=[]
		
		for k in self._points[1:]:
			
			l=self.nearest_nonzero_idx_v2(labels,k[0],k[1])
			
			label=labels[l[0][1]][l[0][0]]
				
			try:
				index=list_of_labels.index(label)
				indexes.append(len(roadwidth))
			except:
				print("except")
				indexes.append(len(roadwidth))
				sumtotal=0
				number=0
				list_of_labels.append(label)
				for teller1,k in enumerate(labels):
					for teller2,l in enumerate(k):
						
						if l==label:
							if logger[teller1][teller2]>2:
								"""
								sumtotal+=logger[teller1][teller2]
								number+=1
			if sumtotal>0:
				list_of_lanes.append(sumtotal/number)
				list_of_labels.append(label)
		
			
								roadwidth.append(sumtotal/number)
								"""
								roadwidth.append(logger[teller1][teller2])
								print(logger[teller1][teller2])
				
		
		return after,roadwidth,indexes

