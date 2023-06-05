#from PIL import Image
#import matplotlib
#matplotlib.use('pdf')
import numpy as np
from numpy import ma
import matplotlib.pyplot as plt
from skimage import transform
from PIL import Image, ImageDraw
import os

def save_results(img, gt, prediction, filename):

    rgb_prediction = img.copy()
    rgb_prediction[gt,0] = 255
    rgb_prediction[gt,1] = 0
    rgb_prediction[gt,2] = 0

    rgb_prediction[prediction,0] = 0
    rgb_prediction[prediction,1] = 0
    rgb_prediction[prediction,2] = 255

    rgb_prediction[np.logical_and(prediction, gt),0] = 255
    rgb_prediction[np.logical_and(prediction, gt),1] = 0
    rgb_prediction[np.logical_and(prediction, gt),2] = 255
    
    rgb_prediction[:,:,1] = np.minimum(rgb_prediction[:,:,1] + prediction*150, 255)
    Image.fromarray(rgb_prediction.astype('uint8')).save(filename + '_gt_pred.png')

    rgb_prediction = img.copy()
    #rgb_prediction[:,:,0] = np.minimum(rgb_prediction[:,:,0] + gt*150, 255)
    rgb_prediction[gt,0] = 255
    rgb_prediction[gt,1] = 0
    rgb_prediction[gt,2] = 0
    Image.fromarray(rgb_prediction.astype('uint8')).save(filename + '_gt.png')

    rgb_prediction = img.copy()
    #rgb_prediction[:,:,1] = np.minimum(rgb_prediction[:,:,1] + prediction*150, 255)
    rgb_prediction[prediction,0] = 255
    rgb_prediction[prediction,1] = 0
    rgb_prediction[prediction,2] = 0
    Image.fromarray(rgb_prediction.astype('uint8')).save(filename + '_pred.png')

def save_tp_fp_fn(img, ltp, lfp, lfn, filename, tp_color=(0,255,0), fp_color=(255,0,0), fn_color=(0,0,255)):
    rgb_prediction = img.copy()
    h = len(img)
    for p in ltp:
        x, y = p
        rgb_prediction[x, y, :] = tp_color
    
        if (x+1 < h):
            rgb_prediction[x+1, y, :] = tp_color
            if(y+1 < h):
                rgb_prediction[x+1, y+1, :] = tp_color
            if(y-1 >= 0):
                rgb_prediction[x+1, y-1, :] = tp_color
                
        if (y+1 < h):
            rgb_prediction[x, y+1, :] = tp_color
        if(x-1 >= 0):
            rgb_prediction[x-1, y, :] = tp_color
            if(y-1 >= 0):
                rgb_prediction[x-1, y-1, :] = tp_color
            if(y+1 < h):
                rgb_prediction[x-1, y+1, :] = tp_color
        if(y-1 >= 0):
            rgb_prediction[x, y-1, :] = tp_color
    
    for p in lfp:
        x, y = p
        rgb_prediction[x, y, :] = fp_color

        if (x+1 < h):
            rgb_prediction[x+1, y, :] = fp_color
            if(y+1 < h):
                rgb_prediction[x+1, y+1, :] = fp_color
            if(y-1 >= 0):
                rgb_prediction[x+1, y-1, :] = fp_color
                
        if (y+1 < h):
            rgb_prediction[x, y+1, :] = fp_color
        if(x-1 >= 0):
            rgb_prediction[x-1, y, :] = fp_color
            if(y-1 >= 0):
                rgb_prediction[x-1, y-1, :] = fp_color
            if(y+1 < h):
                rgb_prediction[x-1, y+1, :] = fp_color
        if(y-1 >= 0):
            rgb_prediction[x, y-1, :] = fp_color
    for p in lfn:
        x, y = p
        rgb_prediction[x, y, :] = fn_color

        if (x+1 < h):
            rgb_prediction[x+1, y, :] = fn_color
            if(y+1 < h):
                rgb_prediction[x+1, y+1, :] = fn_color
            if(y-1 >= 0):
                rgb_prediction[x+1, y-1, :] = fn_color
                
        if (y+1 < h):
            rgb_prediction[x, y+1, :] = fn_color
        if(x-1 >= 0):
            rgb_prediction[x-1, y, :] = fn_color
            if(y-1 >= 0):
                rgb_prediction[x-1, y-1, :] = fn_color
            if(y+1 < h):
                rgb_prediction[x-1, y+1, :] = fn_color
        if(y-1 >= 0):
            rgb_prediction[x, y-1, :] = fn_color
    Image.fromarray(rgb_prediction.astype('uint8')).save(filename + '_tp_fp_fn.png')

def plot_quiver(image, score_mid_t, scale=5, step=8, save=None, axis='off'):
    w = score_mid_t.shape[0]
    h = score_mid_t.shape[1]

    X = np.arange(0, w)
    Y = np.arange(0, h)
    X, Y = np.meshgrid(X, Y)
    U = score_mid_t[:,:,0]*scale
    V = score_mid_t[:,:,1]*scale

    #plt.figure()
    plt.clf()
    plt.axis("off")
    fig = plt.imshow(image)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    fig = plt.quiver(X[::step,::step], Y[::step,::step], U[::step,::step], V[::step,::step], angles='xy', scale=100.)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(save, bbox_inches='tight', pad_inches=0, format='png', dpi=300)
    #plt.savefig(save, bbox_inches='tight', pad_inches=0, format='png', dpi=300)



def save_heatmap(image, heat_map, x, y, save, alpha=0.35, cmap='hot'):
    #plt.figure()
    plt.clf()
    plt.axis('off')
    height = image.shape[0]
    width = image.shape[1]

    # normalize heat map
    max_value = np.max(heat_map)
    min_value = np.min(heat_map)
    normalized_heat_map = (heat_map - min_value) / (max_value - min_value)

    normalized_heat_map = transform.resize(normalized_heat_map, (height, width))
    # display
    fig = plt.imshow(image)
    plt.imshow(255 * normalized_heat_map, alpha=alpha, cmap=cmap)
    if x is not None:
        fig = plt.scatter(y, x, s=10, c='red', marker='o')# s=10
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(save, bbox_inches='tight', pad_inches=0, format='png', dpi=300)
