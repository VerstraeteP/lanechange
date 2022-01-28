import gpxpy 
import gpxpy.gpx 

import math
from PIL import ImageDraw
import argparse
from random import randint
from time import sleep

class gpx2data:
  def __init__(self, points):
    self._points= points
    
  def getXY(self,lat,lng,zoom):				#calculate the corresponding x and y positions of the course points(float)
	
    tile_size = 256
    numTiles = 1 << zoom
    point_x = (tile_size / 2 + lng * tile_size / 360.0) * numTiles / tile_size
    sin_y = math.sin(lat * (math.pi / 180.0))
    point_y = ((tile_size / 2) + 0.5 * math.log((1 + sin_y) / (1 - sin_y)) * -(
    tile_size / (2 * math.pi))) * numTiles / tile_size
    return point_x, point_y
  
  def calculate_points(self):
	
    XY_array=[]
    start_x=-1
    start_y=-1
    new_xy=[]
    for k in self._points:
      
      x1,y1= self.getXY(k[0],k[1],17)				#calculate the course point position(float) in google maps tiles
      if int(x1) not in range(start_x,start_x+3)  or int(y1) not in range(start_y,start_y+3) or start_x==-1 or start_y== -1:		#if tile-coordinate is not already present in a previous extracted 3x3-tile or if it is the first tile
        if len(new_xy)!=0:			#if length of new_xy is diff. of zero, append it to overall array
          XY_array.append(new_xy)
        new_xy=[]
        new_xy.append([int(x1),int(y1)])	#first of all add the google maps tile coordinates to the array
        start_x=int(x1)-1			#get the x-start-coordinate, the top left coordinate of the 3x3-tiles
        start_y=int(y1)-1
        x1-=start_x
        y1-=start_y
        x1*=256
        y1*=256
        new_xy.append([x1,y1])
      else:					#if the tile is already present, calculate the x-y position on the tile according the lat and lon
          x1-=start_x
          y1-=start_y
          x1*=256
          y1*=256
          new_xy.append([x1,y1])		#add position to array
    XY_array.append(new_xy)
    return XY_array
	
		
	

