from skimage.morphology import medial_axis, skeletonize
from skimage import morphology, filters
import cv2
import matplotlib.pyplot as plt
import numpy as np
class skeleton:
	def __init__(self, img, points,tel):
		self._points= points
		self._img= img
		self._tel=tel
		
		
	def nearest_nonzero_idx_v2(self,a,x,y):
	    tmp = a[x,y]
	    a[x,y] = 0
	    r,c = np.nonzero(a)
	    a[x,y] = tmp
	    min_idx = ((r - x)**2 + (c - y)**2).argmin()
	    return r[min_idx], c[min_idx]


	def skeletonization(self):




		roadwidth=[]
		print(self._img.shape)
		self._img = np.uint8(self._img)
		

		des = cv2.bitwise_not(self._img)
		contour,hier = cv2.findContours(des,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
		print(len(contour))

		for cnt in contour:
			cv2.drawContours(des,[cnt],0,255,-1)
		cv2.imwrite("after"+str(self._tel)+".jpg",des)


		self._img = cv2.bitwise_not(des)
		cv2.imwrite("after"+str(self._tel)+".jpg",self._img)
		binary = self._img > filters.threshold_otsu(self._img)
		
		np.unique(binary)
		skel, distance = medial_axis(binary, return_distance=True)
		dist_on_skel = distance * skel
		colormap = plt.get_cmap('magma')
		heatmap = (colormap(dist_on_skel) * 2**16).astype(np.uint16)[:,:,:3]
		heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
		cv2.imwrite("skels"+str(self._tel)+".png",heatmap)
		for k in self._points[1:]:
			
			x, y= self.nearest_nonzero_idx_v2(dist_on_skel,int(k[0]),int(k[1]))

			roadwidth.append(dist_on_skel[x][y])
			
		return roadwidth


