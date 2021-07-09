
import matplotlib.pyplot as plt
import cv2
class crop:
	def __init__(self, img,  points):
		self._points= points
		self._img=img
		
	def max(self,pointlist):
		return max(pointlist)
		
	def min(self, pointlist):
		return min(pointlist)	
	def cropimage(self):
		x=[]
		y=[]
		for k in self._points[1:]:
			x.append(k[0])
			y.append(k[1])
		max_x=self.max(x)
		min_x=self.min(x)
		max_y=self.max(y)
		min_y=self.min(y)
		#only one point in the image
		
		if max_x==min_x:
			max_x+=50
			min_x-=50
			max_y+=50
			min_y-=50
		else:
			min_x-=50
			max_x+=50
			min_y-=50
			max_y+=50
		#check if values don't exceed the borders of the image
		if min_x <0:
			min_x=0
		if min_y<0:
			min_y=0
		if max_x>256*3:
			max_x=256*3
		if max_y>256*3:
			max_y=256*3
		x= max_x-min_x
		y= max_y-min_y
		if x<y:
			diff= y-x
			min_x-=diff/2
			max_x+=diff/2
		if x>y:
			diff= x-y
			min_y-=diff/2
			max_y+=diff/2	
		
		for k in range(len(self._points)-1):
			self._points[k+1]=[self._points[k+1][0]-int(min_x),self._points[k+1][1]-int(min_y)]
		
			
		return self._points,self._img[int(min_y):int(max_y),int(min_x):int(max_x)]
			 
