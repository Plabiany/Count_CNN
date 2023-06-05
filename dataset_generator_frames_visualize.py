from dataset_generator_frames import ImageHeatmapGenerator
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from time import time
from visualize import add, plot_vector_maps

#Tensor("Mconv5_stage1_L1_F0/BiasAdd:0", shape=(?, 64, 64, 2), dtype=float32)
#Tensor("Mconv5_stage1_L2_F0/BiasAdd:0", shape=(?, 64, 64, 1), dtype=float32)
#Tensor("Mconv7_stage2_L1_F0/BiasAdd:0", shape=(?, 64, 64, 2), dtype=float32)
#Tensor("Mconv7_stage2_L2_F0/BiasAdd:0", shape=(?, 64, 64, 1), dtype=float32)
#Tensor("Mconv5_stage1_L1_F1/BiasAdd:0", shape=(?, 64, 64, 2), dtype=float32)
#Tensor("Mconv5_stage1_L2_F1/BiasAdd:0", shape=(?, 64, 64, 1), dtype=float32)
#Tensor("Mconv7_stage2_L1_F1/BiasAdd:0", shape=(?, 64, 64, 2), dtype=float32)
#Tensor("Mconv7_stage2_L2_F1/BiasAdd:0", shape=(?, 64, 64, 1), dtype=float32)
#Tensor("Mconv5_stage1_L1_F2/BiasAdd:0", shape=(?, 64, 64, 2), dtype=float32)
#Tensor("Mconv5_stage1_L2_F2/BiasAdd:0", shape=(?, 64, 64, 1), dtype=float32)
#Tensor("Mconv7_stage2_L1_F2/BiasAdd:0", shape=(?, 64, 64, 2), dtype=float32)
#Tensor("Mconv7_stage2_L2_F2/BiasAdd:0", shape=(?, 64, 64, 1), dtype=float32)

path_rgb = '/Users/wnunes/Dropbox/count/video_01/frames/'
path_labels = '/Users/wnunes/Dropbox/count/video_01/frames/annotations.txt'
g = ImageHeatmapGenerator(path_rgb, path_labels)
n_frames = 3
stages = 2
a = g.generator(5, 3.0, stages, n_frames, 512, 512, 64, 64)
for i in range(50):
	t = time()
	out = next(a)
	print(time()-t)

X = out[0]
Y = out[1]
Z = out[2]

batch = 0
for i in range(len(X)):
	img = Image.fromarray(X[i][batch].astype(np.uint8))
	img.save('frame' + str(i) + '.png')

for i in range(len(Y)):
	add(X[i/stages][batch].astype(np.uint8), Y[i][batch,:,:,0], alpha = 0.35, save='frame_heatmap_' + str(i) + '.png')

for i in range(len(Y)):
	plot_vector_maps(X[i/stages][batch].astype(np.uint8), Z[i][batch], save='frame_vectors_' + str(i) + '.png')