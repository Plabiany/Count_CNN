from numpy import ma
from dataset_generator_utils import get_heatmap, get_edt, get_voronoi
import numpy as np
import os
from PIL import Image
import pickle
from math import sqrt, isnan
import random
import matplotlib

import matplotlib.pyplot as plt

def get_pos(fishes, frame):
  positions = []
  for i in range(len(fishes)):
    fish = fishes[i]
    for j in range(len(fish)):
      if fish[j][0] == frame:
        positions.append([i, fish[j][1], fish[j][2]])
  return positions

def distances(X, Y, x1, y1, x2, y2):
    # classic formula is:
    # d = (x2-x1)*(y1-y)-(x1-x)*(y2-y1)/sqrt((x2-x1)**2 + (y2-y1)**2)
    xD = (x2-x1)
    yD = (y2-y1)
    norm2 = sqrt(xD**2 + yD**2)
    dist = xD*(y1-Y)-(x1-X)*yD
    dist /= norm2
    return np.abs(dist)

def get_vectors(ind, x, y, ind_previous, x_previous, y_previous, width, height, output_width, output_height, thre=1):
  count = np.zeros((output_width, output_height), dtype=np.int)
  heatmaps = np.zeros((output_width, output_height, 2), dtype=np.float64)
  fx = output_width / float(width)
  fy = output_height / float(height)
  for i in range(len(ind_previous)):
    ind_p = ind_previous[i]
    ind_a = -1
    for j,a in enumerate(ind):
      if a == ind_p:
        ind_a = j
        break
    if ind_a == -1:
      continue
    x1 = (x_previous[i]*fx)
    y1 = (y_previous[i]*fy)
    x2 = (x[ind_a]*fx)
    y2 = (y[ind_a]*fy)
    heatmaps, count = get_vector(heatmaps, count, x1, y1, x2, y2, output_width, output_height, thre)
  #heatmaps /= count
  return heatmaps

def get_vector(heatmaps, count, x1, y1, x2, y2, output_width, output_height, thre):
  Y, X = np.mgrid[0:output_height, 0:output_width]
  dx = x2-x1
  dy = y2-y1
  dnorm = sqrt(dx*dx + dy*dy)
  if dnorm==0:
    #print("Points are too close to each other. Length is zero. Skipping")
    return heatmaps, count
  dx = dx / dnorm
  dy = dy / dnorm
  assert not isnan(dx) and not isnan(dy), "dnorm is zero"
  min_sx, max_sx = (x1, x2) if x1 < x2 else (x2, x1)
  min_sy, max_sy = (y1, y2) if y1 < y2 else (y2, y1)
  min_sx = int(round((min_sx - thre)))
  min_sy = int(round((min_sy - thre)))
  max_sx = int(round((max_sx + thre)))
  max_sy = int(round((max_sy + thre)))
  if max_sy < 0 or max_sx < 0:
    return
  if min_sx < 0:
    min_sx = 0
  if min_sy < 0:
    min_sy = 0
  slice_x = slice(min_sx, max_sx)
  slice_y = slice(min_sy, max_sy)
  dist = distances(X[slice_y,slice_x], Y[slice_y,slice_x], x1, y1, x2, y2)
  dist = dist <= thre
  heatmaps[slice_y, slice_x, 0][dist] += (dist * dx)[dist]  # += dist * dx
  heatmaps[slice_y, slice_x, 1][dist] += (dist * dy)[dist]  # += dist * dy
  count[slice_y, slice_x][dist] += 1  
  return heatmaps, count

#path_rgb = '/Users/wnunes/Dropbox/count/video_01/rgb/'
#path_labels = '/Users/wnunes/Dropbox/count/video_01/annotations.txt'

class ImageHeatmapGenerator:
  def __init__(self, path_rgb, extension='png', sort_frames=True):
    self.image_names = []
    self.path_rgb = []
    self.path_labels = []
    self.n_images = 0
    self.extension = extension
    self.videos_ = os.listdir(path_rgb)
    self.videos_.sort()

    for r in range(len(self.videos_)):
      #self.path_rgb.append(os.path.join(path_rgb,self.videos_[r] + '/frames'))
      #self.path_labels.append(os.path.join(path_rgb,self.videos_[r] + '/frames/annotations.txt'))
      self.path_rgb.append(os.path.join(path_rgb,self.videos_[r]))
      self.path_labels.append(os.path.join(path_rgb,self.videos_[r]+'/annotations.txt'))
    
    for p in range(len(self.path_rgb)):
      aux_ = [f for f in os.listdir(self.path_rgb[p]) if f.endswith(self.extension)]
      if sort_frames:
        aux_.sort()
      self.image_names.append(aux_)
      self.n_images += len(aux_)
      #self.image_names.sort()

  # imgs_ = (batch_size, width, height, 3*n_frames)
  # labels = [..., L(t-1,0), ..., L(t-1,s), L(t,0), ..., L(t,s)], L(t,s) = (batch_size, output_width, output_height, 1)
  def generator(self, batch_size, sigma, stages, n_frames, width, height, temporal, output_width, output_height, min_sigma, is_sequential, square):
    pos = n_frames
    vid = 0
    fishes = []
    for p in range(len(self.path_labels)):
      with open(self.path_labels[p], "rb") as file:
        fishes.append(pickle.load(file))  
    sigmas = np.linspace(sigma,min_sigma,stages)
    while True:
      labels = []
      self.x_points = []
      self.y_points = []
      self.indices = []
      self.filenames = []
      imgs = np.zeros((batch_size, width, height, 3*n_frames))
      labels = np.zeros((stages,batch_size, output_width, output_height, 1*n_frames))
      #labels_edt = np.zeros((stages,batch_size, output_width, output_height, 1*n_frames))
      labels_vectors = np.zeros((stages,batch_size, output_width, output_height, 2*n_frames))
    
      for i in range(batch_size):
        filenames_frames = []
        x_frames = []
        y_frames = []
        ind_frames = []
        for f in range(n_frames):
          frame = pos-(n_frames-1)+f
          img = Image.open(os.path.join(self.path_rgb[vid], self.image_names[vid][frame]))
          
          filenames_frames.append(os.path.join(self.path_rgb[vid], self.image_names[vid][frame]))
          fx = (float(width) / float(img.width))
          fy = (float(height) / float(img.height))
          
          imgs[i,:,:,f*3:(f+1)*3] = np.array(img.resize((width,height)))
          pos_fishes = get_pos(fishes[vid], frame)
          pos_fishes_previous = get_pos(fishes[vid], frame-1)
          
          ind_previous = [p[0] for p in pos_fishes_previous]
          x_previous = [p[1]*fx for p in pos_fishes_previous]
          y_previous = [p[2]*fy for p in pos_fishes_previous]  
          
          ind = [p[0] for p in pos_fishes]
          x = [p[1]*fx for p in pos_fishes]
          y = [p[2]*fy for p in pos_fishes]
          x_frames.append(x)
          y_frames.append(y)
          ind_frames.append(ind)
          
          for si, sig in enumerate(sigmas):
            if square:
              label, _ = get_heatmap(x, y, width, height, 1, output_width, output_height, square)
            else:
              label, _ = get_heatmap(x, y, width, height, sig, output_width, output_height, square)

            #label_edt = get_voronoi(x, y, width, height, output_width, output_height)
            #yv, xv = np.where(label_edt > 0)
            #label_edt, _ = get_heatmap(xv, yv, output_width, output_height, 0.25, output_width, output_height, square)

            labels[si,i,:,:,f] = label
            #labels_edt[si,i,:,:,f] = label_edt
            label_vector = get_vectors(ind, x, y, ind_previous, x_previous, y_previous, width, height, output_width, output_height)
            labels_vectors[si,i,:,:,f*2:(f+1)*2] = label_vector
          
        self.x_points.append(x_frames)
        self.y_points.append(y_frames)
        self.filenames.append(filenames_frames)
        self.indices.append(ind_frames)

        if is_sequential:
          pos = max((pos+1) % self.n_images, n_frames)
          if pos % len(self.image_names[vid]) == 0:
            if vid == len(self.path_rgb)-1:
              vid = 0
            else:
              vid+=1
            pos = n_frames
        else:
          vid = random.randint(0,len(self.videos_)-1)
          pos = random.randint(n_frames, len(self.image_names[vid])-1)

      labels_all = []
      for i in range(stages):
        if temporal:
          labels_all.append(labels_vectors[si,:,:,:,(2*(n_frames-1)):2*n_frames])
        #labels_all.append(labels_edt[si])
        labels_all.append(labels[si,:,:,:,-1])

      yield imgs, labels_all