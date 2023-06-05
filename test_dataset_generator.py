from model_frames import get_training_model
from dataset_generator_frames2 import ImageHeatmapGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from visualize import save_heatmap, plot_quiver
from skimage import transform
from skimage.feature import peak_local_max

num_steps = 1000
batch_size = 1
sigma = 2.
min_sigma = 1.
stages = 4
square = None
n_frames = 3
img_w = 512
img_h = 512
output_width = 128
output_height = 128
train_path_imgs = '/home/wnunes/Plabiany/Dataset_Alevinos/Train'
pathout = 'temp'
alpha = 0.5
min_distance = 2
thre1 = 0.15
is_sequential = True

train_imageHeatmapGenerator = ImageHeatmapGenerator(train_path_imgs)
train_generator = train_imageHeatmapGenerator.generator(batch_size, sigma, stages, n_frames, img_w, img_h, output_width=output_width, output_height=output_height, min_sigma=min_sigma, is_sequential=is_sequential, square = square)

if not os.path.isdir(pathout):
    os.makedirs(pathout)

for i in range(90):
	print(i)
	oriImg, output_blobs = next(train_generator)

for i in range(num_steps):
	oriImg, output_blobs = next(train_generator)
	heatmaps = output_blobs[-1]
	vectors = output_blobs[-2]
		
	for b in range(batch_size):
		filenames = train_imageHeatmapGenerator.filenames[b]

		imgs = oriImg[b]
		heatmap = heatmaps[b]
		vector = vectors[b]

		f = n_frames-1
		img = imgs[:,:,f*3:(f+1)*3]
		
		#img_line = transform.resize(line, (img.shape[0], img.shape[1]))

		img_hm = transform.resize(heatmap, (img.shape[0], img.shape[1]))
		peaks = peak_local_max(img_hm, min_distance=min_distance, threshold_abs=thre1, exclude_border=False)
		x1 = [a[0] for a in peaks]
		y1 = [a[1] for a in peaks]

		img_vectors = vector
		img_vectors = transform.resize(img_vectors, (img.shape[0], img.shape[1]))
		
		filename_img_frame = os.path.basename(filenames[f])
		#Image.fromarray(img.astype(np.uint8)).save(os.path.join(pathout, filename_img_frame))
		#save_heatmap(img.astype(np.uint8), img_line, None, None, os.path.join(pathout, filename_img_frame[:-4] + '_lines' + '.png'), alpha=alpha)
		save_heatmap(img.astype(np.uint8), img_hm, None, None, os.path.join(pathout, filename_img_frame[:-4] + '_points_' + '.png'), alpha=alpha)			
		plot_quiver(img.astype(np.uint8), img_vectors, save=os.path.join(pathout, filename_img_frame[:-4]+'_vectors' + '.png'), scale=5, step=5)