from skimage.morphology import medial_axis, skeletonize
from skimage import morphology, filters
import cv2
import matplotlib.pyplot as plt
import numpy as np
class skeleton:
	def __init__(self, img, points):
		self._points= points
		self._img= img
		
		
	def nearest_nonzero_idx_v2(a,x,y):
	    tmp = a[x,y]
	    a[x,y] = 0
	    r,c = np.nonzero(a)
	    a[x,y] = tmp
	    min_idx = ((r - x)**2 + (c - y)**2).argmin()
	    return r[min_idx], c[min_idx]


	def skeletonization(self):




		roadwidth=[]
		binary = img > filters.threshold_otsu(self._img)
		np.unique(binary)
		skel, distance = medial_axis(binary, return_distance=True)
		dist_on_skel = distance * skel
		for k in self._points[1:]:
			
			x, y= nearest_nonzero_idx_v2(dist_on_skel,int(k[0]),int(k[1]))

			roadwidth.append(dist_on_skel[x][y]))
			
		return roadwidth


