import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage.filters import gaussian_filter
from keras.models import load_model
from visualize import add
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from dataset_generator import ImageHeatmapGenerator
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from model import Interp
from metrics import oks
import math

def RSS(o, p):
    return np.sum((o - p) ** 2)

def rmsd(x, y, ddof=0):
    return np.sqrt(RSS(x, y) / (len(x) - ddof))

def nrmsd(x, y, ddof=0):
    x = np.array(x)
    y = np.array(y)
    return rmsd(x, y, ddof) / (np.max([x, y]) - np.min([x, y]))

#stages = 4
#sigma = 1.5
#thre1 = 0.35
#use_filter = False
#min_distance = 1
#model_path = '/home/wnunes/datasets/count_plants/count_plants_vgg19_psp1236_sigma_2_0_minsigma_2_0_stages_4/model_final.h5'
#path = '/home/wnunes/datasets/count_plants/test/rgb'
#pathLabels = '/home/wnunes/datasets/count_plants/test/labels/'

#stages = 4
#sigma = 1.5
#thre1 = 0.15
#min_distance = 1
#use_filter = False
#model_path = '/home/lavicom/datasets/carpk_pucpr/datasets/CARPK_devkit/carpk_vgg19_psp1236_sigma_3_0_minsigma_1_0_stages_4_da/model_final.h5'
#path = '/home/lavicom/datasets/carpk_pucpr/datasets/CARPK_devkit/data/test_1024x1024'
#pathLabels = '/home/lavicom/datasets/carpk_pucpr/datasets/CARPK_devkit/data/test_point'

stages = 4
sigma = 2.0
thre1 = 0.15
min_distance = 1
use_filter = False
model_path = '/home/lavicom/datasets/carpk_pucpr/datasets/CARPK_devkit/pucpr_vgg19_psp1236_sigma_2_0_minsigma_0_5_stages_4_da/model_final.h5'
path = '/home/lavicom/datasets/carpk_pucpr/datasets/PUCPR+_devkit/data/test_1024x1024'
pathLabels = '/home/lavicom/datasets/carpk_pucpr/datasets/PUCPR+_devkit/data/test_point'


extension = 'png'

alpha = 0.35

model = load_model(model_path, custom_objects={'Interp': Interp})
output_width = output_height = model.output[0].get_shape()[1].value

imageHeatmapGenerator = ImageHeatmapGenerator(path, pathLabels)
generator = imageHeatmapGenerator.generator(1, sigma, stages, output_width=output_width, output_height=output_height)

pred = []
real = []
anno_x = []
anno_y = []
pred_x = []
pred_y = []
filenames = []

for i in range(imageHeatmapGenerator.n_images):
#for i in range(10):

    oriImg, label = next(generator)
    filename = imageHeatmapGenerator.filenames[0]
    filenames.append(filename)
    #oriImg = np.array(Image.open(os.path.join(path, filename)))

    output_blobs = model.predict(oriImg)

    heatmap = output_blobs[-1]
    heatmap = heatmap[0,:,:,0]

    all_peaks = []
    peak_counter = 0

    #map = gaussian_filter(heatmap, sigma=sigma)
    #map = heatmap

    #map_left = np.zeros(map.shape)
    #map_left[1:, :] = map[:-1, :]
    #map_right = np.zeros(map.shape)
    #map_right[:-1, :] = map[1:, :]
    #map_up = np.zeros(map.shape)
    #map_up[:, 1:] = map[:, :-1]
    #map_down = np.zeros(map.shape)
    #map_down[:, :-1] = map[:, 1:]

    #peaks_binary = np.logical_and.reduce((map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > thre1))
    #peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
    #fx = heatmap.shape[0] / float(oriImg.shape[1])
    #fy = heatmap.shape[1] / float(oriImg.shape[2])
    #x = [a[0]/fx for a in peaks]
    #y = [a[1]/fy for a in peaks]
    #peak_counter = len(peaks)
    #add(oriImg[0,:,:,:], heatmap, x=y, y=x, alpha=alpha, save=filename + '_predict.png', axis='off')

    if use_filter:
        heatmap = gaussian_filter(heatmap, sigma=sigma)
    peaks = peak_local_max(heatmap, min_distance=min_distance, threshold_abs=thre1, exclude_border=False)
    fx = heatmap.shape[0] / float(oriImg.shape[1])
    fy = heatmap.shape[1] / float(oriImg.shape[2])
    x = [a[0]/fx for a in peaks]
    y = [a[1]/fy for a in peaks]
    peak_counter = len(peaks)
    add(oriImg[0,:,:,:], heatmap, x=x, y=y, alpha=alpha, save=filename + '_predict.png', axis='off')    
    pred_x.append(x)
    pred_y.append(y)

    #img_label = np.array(Image.open(os.path.join(pathLabels, filename)))
    x = imageHeatmapGenerator.x_points[0]
    y = imageHeatmapGenerator.y_points[0]
    heatmap_label = label[-1]
    heatmap_label = heatmap_label[0,:,:,0]
    anno_x.append(x)
    anno_y.append(y)

    #plt.imshow(heatmap, cmap='hot')
    #plt.show()

    #heat_map = get_heamap(y, x, img_label.shape[0], img_label.shape[1], 1., 64, 64)
    add(oriImg[0,:,:,:], heatmap_label, x=x, y=y, alpha=alpha, save=filename + '_label.png', axis='off')

    print('%d (%s) of %d: %d - %d\n' % (i, filename, imageHeatmapGenerator.n_images, peak_counter, len(x)))

    pred.append(peak_counter)
    real.append(len(x))

print(' MAE: %f\n' % mean_absolute_error(real, pred))
print(' MSE: %f\n' % mean_squared_error(real, pred))
print(' RMSE: %f\n' % math.sqrt(mean_squared_error(real, pred)))
print(' R2: %f\n' % r2_score(real, pred))
print(' NRMSE: %f\n' % nrmsd(real, pred))
oks_m = oks(anno_x, anno_y, pred_x, pred_y)
print(' OKS: %f\n' % oks_m[1][0])

np.save('anno_x', anno_x)
np.save('anno_y', anno_y)
np.save('pred_x', pred_x)
np.save('pred_y', pred_y)
np.save('filenames', filenames)
