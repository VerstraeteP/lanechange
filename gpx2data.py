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
    
  def getXY(self,lat,lng,zoom):
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
      
      x1,y1= self.getXY(k[0],k[1],17)
      if int(x1) not in range(start_x,start_x+3)  or int(y1) not in range(start_y,start_y+3) or start_x==-1 or start_y== -1:
        if len(new_xy)!=0:
          XY_array.append(new_xy)
        new_xy=[]
        new_xy.append([int(x1),int(y1)])
        start_x=int(x1)-1
        start_y=int(y1)-1
        x1-=start_x
        y1-=start_y
        x1*=256
        y1*=256
        new_xy.append([x1,y1])
      else:
          x1-=start_x
          y1-=start_y
          x1*=256
          y1*=256
          new_xy.append([x1,y1])
    return XY_array
		
	

