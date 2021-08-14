#!/usr/bin/python
import sys
from scipy import ndimage
import cv2
import pandas as pd
import numpy as np
import heapq as hq 
import time
from PIL import Image
from random import random

def random_color():
    return (int(random()*255), int(random()*255), int(random()*255))
    
def create_segemented_image(width, height, region):
    colors = [random_color() for i in range(11)]
    
    img = Image.new('RGB', (width, height))
    image = img.load()
    for y in range(height):
        for x in range(width):
            label = region[x,y]
            image[x, y] = colors[int(label)+1 ]

    return img.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT)
    
def get_neighbors(height, width, x, y, region, h, flag, grad):
    if x-1 >= 0 and region[x-1,y] == 0:
        hq.heappush(h, (grad[x-1,y], (x-1, y)))
        flag.add((x-1, y))
        
    if y-1 >= 0 and region[x,y-1] == 0:
        hq.heappush(h, (grad[x,y-1], (x,y-1)))
        flag.add((x,y-1))
        
    if x+1 < width and region[x+1,y] == 0:
        hq.heappush(h, (grad[x+1,y], (x+1,y)))
        flag.add((x+1,y))
        
    if y+1 < height and region[x,y+1] == 0:
        hq.heappush(h, (grad[x,y+1], (x,y+1)))
        flag.add((x,y+1))
    
    
def get_neighbors_label(height, width, x, y, region, h, flag, grad):
    
    l = []
    if x-1 >= 0:
        if region[x-1,y] == 0 and not ((x-1, y) in flag):
            hq.heappush(h, (grad[x-1,y], (x-1, y)))
            flag.add((x-1, y))
              
        elif region[x-1,y] != 0:
            l.append(region[x-1,y])
        
    if y-1 >= 0:
        if region[x,y-1] == 0 and not ((x, y-1) in flag):
            hq.heappush(h, (grad[x,y-1], (x, y-1)))
            flag.add((x,y-1))
            
        elif region[x,y-1] != 0:
            l.append(region[x,y-1])
            
        
    if x+1 < width:
        if region[x+1,y] == 0 and not ((x+1, y) in flag):
            hq.heappush(h, (grad[x+1,y], (x+1, y)))
            flag.add((x+1,y))
            
        elif region[x+1,y] != 0:
            l.append(region[x+1,y])
        
    if y+1 < height:
        if region[x,y+1] == 0 and not ((x, y+1) in flag):
            hq.heappush(h, (grad[x,y+1], (x, y+1)))
            flag.add((x,y+1))
            
        elif region[x,y+1] != 0:
            l.append(region[x,y+1])
  
   #create label for given pixel
    if l.count(l[0]) == len(l):
        region[x,y] = l[0]
 
    else:
        region[x,y] = -1


def main():
  if len(sys.argv) < 5:
    print ('Usage: wshedSegment.py inputImageFile inputSeedFile outputImageFile sigma')
    sys.exit(1)

  inputImageFile = sys.argv[1]
  inputSeedFile = sys.argv[2]
  outputImageFile = sys.argv[3]
  sigma = float(sys.argv[4])
  
  start_time = time.time()
  img = cv2.imread(inputImageFile,0)
  seeds = pd.read_csv(inputSeedFile, header=None, delim_whitespace=True)
  
  grad = ndimage.gaussian_gradient_magnitude(img, sigma=sigma)
  region = np.zeros(img.shape)
  h = [] #priority queue
  flag = {*()} #use a set to make sure pixel goes in priority queue only once
  height = img.shape[0]
  width = img.shape[1]

  for i in range(seeds.shape[0]):
    x = int(seeds.iloc[i,2])
    y = int(seeds.iloc[i,1])
    region[x,y] = int(seeds.iloc[i,0])
    get_neighbors(img.shape[0], img.shape[1], x, y, region, h, flag, grad)
    
  while h:
    pix = hq.heappop(h)
    get_neighbors_label(img.shape[0], img.shape[1], pix[1][0], pix[1][1], region, h, flag, grad)
    
  elapsed_time = time.time() - start_time
    
  print( "Execution time: " + str(int(elapsed_time / 60)) + " minute(s) and " + str(int(elapsed_time % 60)) + " seconds")
  
  output_image = create_segemented_image(width, height, region)
  output_image.save(outputImageFile)

if __name__ == '__main__':
  main()
