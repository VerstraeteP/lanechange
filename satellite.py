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
from scipy.signal import convolve2d
from skan.csr import make_degree_image

import matplotlib.pyplot as plt

class road_map:
	def __init__(self,image):
		self.image=image
	def convolution(self):
		
		kernel = np.ones((5,3),np.uint8)
		#kernel[1,1]=0
		self.image = cv2.erode(self.image, kernel) 
		#mask = convolve2d(self.image, kernel, mode='same', fillvalue=1)
		
		#result = image.copy()
		#result[np.logical_and(mask==8, test==0)] = 1
		cv2.imwrite("neighbours.png",self.image)

		return self.image

	def make_skeleton(self):
		
		lower_white = np.array([254, 254, 254 ])
		upper_white = np.array([255, 255, 255])
		white_range = cv2.inRange(self.image, lower_white,upper_white)
		self.image[white_range!=0] = [255,255,255]
		lower_yellow = np.array([160, 230, 250 ])
		upper_yellow = np.array([178, 249, 260])
		yellow_range = cv2.inRange(self.image, lower_yellow,upper_yellow)
		self.image[yellow_range!=0] = [255,255,255]
		self.image[np.where((self.image!=[255,255,255]).all(axis=2))] = [0,0,0]

		self.image= cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
		self.image[np.where((self.image!=[255]))] = [0]
		kernel = np.int8([[-1, -1, -1],[-1, +1, -1],[-1, -1, -1]])
		neighbors_all_zero = cv2.morphologyEx(src=self.image, op=cv2.MORPH_HITMISS, kernel=kernel)
		self.image = self.image & ~neighbors_all_zero
		
		mask=self.convolution()
		return mask
	def nearest_nonzero_idx_v2(self,a,x,y):
		a.astype(int)
		nonzero = cv2.findNonZero(a)
		
		distances = np.sqrt((nonzero[:,:,0] - x) ** 2 + (nonzero[:,:,1] - y) ** 2)
		nearest_index = np.argmin(distances)
		
					  
		return nonzero[nearest_index]
	

	def skeletonization(self):




		roadwidth=[]
		
		
		self.image = np.uint8(self.image)
		binary = self.image > filters.threshold_otsu(self.image)
		
		np.unique(binary)
		skel, distance = medial_axis(binary, return_distance=True)
		dist_on_skel = distance * skel
	
		#cv2.imwrite("skelssat"+str(self._tel)+".png",heatmap)
		#find crossings 
		degrees = make_degree_image(skel)
		intersection_matrix = degrees > 2
		
		#list with crossings
		lists=np.argwhere(intersection_matrix== True)
		#set crossingpixel to zero
		logger=dist_on_skel.copy()
		cv2.imwrite("neighboursbefore.png",self.image)

		for teller,k in enumerate(dist_on_skel):
			for teller2,l in enumerate(k):
				if int(l)!=0:
		
					dist_on_skel[teller][teller2]=255
    
		return dist_on_skel

