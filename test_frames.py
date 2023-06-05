import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage.filters import gaussian_filter
from keras.models import load_model
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from dataset_generator_frames2 import ImageHeatmapGenerator
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
import math
import cv2
from scipy.optimize import linear_sum_assignment
import argparse
from metrics import precision_recall_f1, confusion_matrix_p, generate_statistics
from skimage import transform
from visualize import save_heatmap, plot_quiver
import tensorflow as tf
#import keras

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def RSS(o, p):
    return np.sum((o - p) ** 2)

def rmsd(x, y, ddof=0):
    return np.sqrt(RSS(x, y) / (len(x) - ddof))

def nrmsd(x, y, ddof=0):
    x = np.array(x)
    y = np.array(y)
    return rmsd(x, y, ddof) / (np.max([x, y]) - np.min([x, y]))


parser = argparse.ArgumentParser(description='Training Counting CNN')
parser.add_argument('-width', action="store", required=True, help='image width', dest='img_w', type=int)
parser.add_argument('-height', action="store", required=True, help='image height', dest='img_h', type=int)
parser.add_argument('-n_fish', action="store", required=True, help='n_fish', dest='n_fish', type=int)
parser.add_argument('-n_frames', action="store", required=True, help='number of frames', dest='n_frames', type=int)
parser.add_argument('-stages', action="store", required=True, help='number of stages', dest='stages', type=int)
parser.add_argument('-sigma', action="store", required=True, help='sigma', dest='sigma', type=float)
parser.add_argument('-min_sigma', action="store", required=True, help='sigma', dest='min_sigma', type=float)
parser.add_argument('-thre1', action="store", required=True, help='sigma', dest='thre1', type=float)
parser.add_argument('-experiment_folder', action="store", required=True, help='folder to save the experiment', dest='experiment_name')
parser.add_argument('-train_images_folder', action="store", required=True, help='training rgb images folder', dest='train_path_imgs')

arguments = parser.parse_args()

model_path = arguments.experiment_name
path = arguments.train_path_imgs
img_w = arguments.img_w
img_h = arguments.img_h
sigma = arguments.sigma
min_sigma = arguments.min_sigma
thre1 = arguments.thre1
stages = arguments.stages
n_frames = arguments.n_frames
n_fish = arguments.n_fish
square = False
temporal = True

min_distance = 10
border = 30

alpha = 0.35

model = load_model(model_path)
output_width = output_height = model.output[0].get_shape()[1]

imageHeatmapGenerator = ImageHeatmapGenerator(path)
generator = imageHeatmapGenerator.generator(1, sigma, stages, n_frames, img_w, img_h, temporal, output_width=output_width, output_height=output_height, min_sigma=min_sigma, is_sequential=True, square = square)

pred = []
real = []
precisions = []
recalls = []
f1s = []
filenames = []

print(model.outputs)

aux_modelpath = model_path.split('/')
path_to_save = ''
for i in range(len(aux_modelpath)-1):
  path_to_save+= aux_modelpath[i] + '/'
#os.mkdir(os.path.join(model_path[:len(model_path)-14],'result'))
#pathout = 'temp'
pathout = path_to_save + 'results'
print("PATHTOSAVE: "+ path_to_save)
if not os.path.isdir(pathout):
    os.makedirs(pathout)

for b in range(imageHeatmapGenerator.n_images):
    oriImg, label = next(generator)
    filenames_all = imageHeatmapGenerator.filenames[0]
    y2 = imageHeatmapGenerator.x_points[0][-1]
    x2 = imageHeatmapGenerator.y_points[0][-1]
    print('%d of %d' % (b, imageHeatmapGenerator.n_images))
    if len(x2) >= n_fish:
        frame_t = oriImg[0,:,:,-3:]
        output_blobs = model.predict(oriImg)

        heatmaps = output_blobs[-1]
        heatmaps_label = label[-1]
        imgs = oriImg[0]
        heatmap = heatmaps[0]

        #for f in range(n_frames):
        f = n_frames-1
        img = imgs[:,:,f*3:(f+1)*3]
        
        img_hm = transform.resize(heatmap[:,:,0], (img.shape[0], img.shape[1]))
        peaks = peak_local_max(img_hm, min_distance=min_distance, threshold_abs=thre1, exclude_border=False)
        x1 = [a[0] for a in peaks]
        y1 = [a[1] for a in peaks]

        filename_img_frame = os.path.basename(filenames_all[f])

        x1_ = [x1[i] for i in range(len(x1)) if x1[i] > border and y1[i] > border and x1[i] <= img.shape[0]-border and y1[i] <= img.shape[1]-border]
        y1_ = [y1[i] for i in range(len(y1)) if x1[i] > border and y1[i] > border and x1[i] <= img.shape[0]-border and y1[i] <= img.shape[1]-border]
        x2_ = [x2[i] for i in range(len(x2)) if x2[i] > border and y2[i] > border and x2[i] <= img.shape[0]-border and y2[i] <= img.shape[1]-border]
        y2_ = [y2[i] for i in range(len(y2)) if x2[i] > border and y2[i] > border and x2[i] <= img.shape[0]-border and y2[i] <= img.shape[1]-border]
        x1, y1 = x1_, y1_
        x2, y2 = x2_, y2_

        img_PIL = Image.fromarray(img.astype(np.uint8)).convert('RGBA')
        ap, ar, f1 = precision_recall_f1(x2, y2, x1, y1, os.path.join(pathout, filename_img_frame[:-4] + '_points.png'), max_distance=20, point_size=3, w=img.shape[0], h=img.shape[1], img=img_PIL)# max_distance=8 (milho) | 15 (Laranjais - 256pixels) | 60 e 5 (Laranjais - 1024 pixels)

        precisions.append(ap)
        recalls.append(ar)
        f1s.append(f1)
        pred.append(len(x1))
        real.append(len(x2))
        filenames.append(filename_img_frame)

        if temporal:
            vectors = output_blobs[-2]
            vector = vectors[0]
            img_vectors = vector[:,:]
            img_vectors = transform.resize(img_vectors, (img.shape[0], img.shape[1]))


        #if b > 100 and b < 110:
        Image.fromarray(img.astype(np.uint8)).save(os.path.join(pathout, filename_img_frame))
        save_heatmap(img.astype(np.uint8), img_hm, x1, y1, os.path.join(pathout, filename_img_frame[:-4] + '_points_' + '.png'), alpha=alpha)           
        save_heatmap(img.astype(np.uint8), img_hm, x2, y2, os.path.join(pathout, filename_img_frame[:-4] + '_points_gt_' + '.png'), alpha=alpha)           
        if temporal:
            plot_quiver(img.astype(np.uint8), img_vectors, save=os.path.join(pathout, filename_img_frame[:-4]+'_vectors' + '.png'), scale=5, step=5)

n_max = 2
results = generate_statistics(filenames, precisions, recalls, f1s, pred, real, n_max)
for n in range(n_max+1):
    if len(results[n]['gt']) == 0 or len(results[n]['pred']) == 0:
      print("N:0")
    else:
      #results[n]['filename']
      results[n]['precision']
      results[n]['recall']
      results[n]['f1']
      results[n]['pred']
      results[n]['gt']
      print(' N: %d\n' % n)
      print(' MAE: %f\n' % mean_absolute_error(results[n]['gt'], results[n]['pred']))
      print(' MSE: %f\n' % mean_squared_error(results[n]['gt'], results[n]['pred']))
      print(' R2: %f\n' % r2_score(results[n]['gt'], results[n]['pred']))
      print(' NRMSE: %f\n' % nrmsd(results[n]['gt'], results[n]['pred']))

      print(' Precision: %f\n' % np.mean(results[n]['precision']))
      print(' Recall: %f\n' % np.mean(results[n]['recall']))
      print(' F1: %f\n' % np.mean(results[n]['f1']))

#print(' OKS: %f\n' % oks(anno_x, anno_y, pred_x, pred_y))
'''
np.save(model_path[:146]+'anno_x.npy', anno_x)
np.save(model_path[:146]+'anno_y.npy', anno_y)
np.save(model_path[:146]+'pred_x.npy', pred_x)
np.save(model_path[:146]+'pred_y.npy', pred_y)
'''