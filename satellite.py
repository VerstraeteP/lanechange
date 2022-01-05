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
	
		cv2.imwrite("skels"+str(self._tel)+".png",heatmap)
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
"""
kernel = np.int8([ # 0 means "don't care", all others have to match
    [-1, -1, -1],
    [-1, +1, -1],
    [-1, -1, -1],
])

neighbors_all_zero = cv2.morphologyEx(src=im, op=cv2.MORPH_HITMISS, kernel=kernel)

im = im & ~neighbors_all_zero
kernel = np.ones((5,5), np.uint8)
 
# The first parameter is the original image,
# kernel is the matrix with which image is
# convolved and third parameter is the number
# of iterations, which will determine how much
# you want to erode/dilate a given image.
im = cv2.erode(im, kernel, iterations=1)
im = cv2.dilate(im, kernel, iterations=1)
plt.imshow(im) 
plt.show()

img=im
plt.imshow(im) 
plt.show()

#-----Converting image to LAB Color model----------------------------------- 
lab= cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
plt.imshow(lab)
plt.show()
#-----Splitting the LAB image to different channels-------------------------
l, a, b = cv2.split(lab)
plt.imshow(l)
plt.show()
plt.imshow(a)
plt.show()
plt.imshow(b)
plt.show()

#-----Applying CLAHE to L-channel-------------------------------------------
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(3,3))
cl = clahe.apply(l)
plt.imshow(cl)
plt.show()

#-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
limg = cv2.merge((cl,a,b))
plt.imshow(limg)
plt.show()

#-----Converting image from LAB Color model to RGB model--------------------
final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
cv2.imwrite("final.png",final)
plt.imshow(final)
plt.show()
points=[]

"""
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
		

		self._img = img
		
		#self._img = cv2.bitwise_not(self._img)
		




		binary = self._img > filters.threshold_otsu(self._img)
		
		np.unique(binary)
		skel, distance = medial_axis(binary, return_distance=True)
		dist_on_skel = distance * skel
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
		ax1.imshow(self._img, cmap=plt.cm.gray, interpolation='nearest')
		ax1.axis('off')
		
		ax2.imshow(dist_on_skel, cmap = plt.cm.get_cmap("Spectral"), interpolation='nearest')
		ax2.contour(self._img, [0.5], colors='w')
		ax2.axis('off')

		fig.tight_layout()
		plt.show()
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
		print("lists")
		print(lists)
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
sk=skeleton(im,points,0)
r,a,b=sk.skeletonization()
