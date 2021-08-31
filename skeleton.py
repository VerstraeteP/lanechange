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
class skeleton:
	def __init__(self, img, points,tel):
		self._points= points
		self._img= img
		self._tel=tel
		
	"""	
	def nearest_nonzero_idx_v2(self,a,x,y):
	    tmp = a[x,y]
	    a[x,y] = 0
	    r,c = np.nonzero(a)
	    a[x,y] = tmp
	    min_idx = ((r - x)**2 + (c - y)**2).argmin()
	    return r[min_idx], c[min_idx]
	"""
	def nearest_nonzero_idx_v2(self,a,x,y):
		a.astype(int)
		nonzero = cv2.findNonZero(a)
		
		distances = np.sqrt((nonzero[:,:,0] - x) ** 2 + (nonzero[:,:,1] - y) ** 2)
		nearest_index = np.argmin(distances)
		
					  
		return nonzero[nearest_index]


	def skeletonization(self):




		roadwidth=[]
		"""
		img = np.uint8(self._img)
		kernel = np.ones((20,20),np.uint8)
		"""
		
		img = np.uint8(self._img)
		kernel = np.ones((10,10),np.uint8)
		after = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

		self._img = after
		cv2.imwrite("after"+str(self._tel)+".jpg",self._img)
		#self._img = cv2.bitwise_not(self._img)
		




		binary = self._img > filters.threshold_otsu(self._img)
		
		np.unique(binary)
		skel, distance = medial_axis(binary, return_distance=True)
		dist_on_skel = distance * skel
		colormap = plt.get_cmap('magma')
		heatmap = (colormap(dist_on_skel) * 2**16).astype(np.uint16)[:,:,:3]
		heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
		cv2.imwrite("skels"+str(self._tel)+".png",heatmap)
		"""
		for k in self._points[1:]:
			print("before:",self._tel,int(k[0]),int(k[1]))
			x, y= self.nearest_nonzero_idx_v2(dist_on_skel,int(k[0]),int(k[1]))
			print("after:",x,y)
			print("width",dist_on_skel[y][x])
		"""
		
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
				roadwidth.append(list_of_lanes[index])
			except:
				print("except")
        
				sumtotal=0
				number=0
				list_of_labels.append(label)
				for teller1,k in enumerate(labels):
					for teller2,l in enumerate(k):
						
						if l==label:
							if logger[teller1][teller2]>2:
								
								sumtotal+=logger[teller1][teller2]
								number+=1
				if sumtotal>0:
				  list_of_lanes.append(sumtotal/number)
				  list_of_labels.append(label)

			

				  roadwidth.append(sumtotal/number)
		
		
		return after,roadwidth


