import numpy as np
import os
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageDraw
from scipy import ndimage

def generate_statistics(filenames, precisions, recalls, f1s, pred, real, n_max = 10):
    results = {}
    for n in range(n_max+1):
        results[n] = {}
        results[n]['filename'] = []
        results[n]['precision'] = []
        results[n]['recall'] = []
        results[n]['f1'] = []
        results[n]['pred'] = []
        results[n]['gt'] = []

    for i in range(len(filenames)):
        n_real = min(real[i], n_max+1)
        for n in range(n_real+1):
            results[n]['filename'].append(filenames[i])
            results[n]['precision'].append(precisions[i])
            results[n]['recall'].append(recalls[i])
            results[n]['f1'].append(f1s[i])
            results[n]['pred'].append(pred[i])
            results[n]['gt'].append(real[i])
    return results



def confusion_matrix_p(img_gt, img_pred, p=0):
    img_pred = img_pred > 0
    img_gt = img_gt > 0
    distance_gt = ndimage.distance_transform_edt(np.logical_not(img_gt))
    distance_pred = ndimage.distance_transform_edt(np.logical_not(img_pred))
    w, h = img_pred.shape
    tp = fn = fp = 0.0
    ltp = []
    lfn = []
    lfp = []
    for x in range(w):
        for y in range(h):
            if img_pred[x,y] and distance_gt[x,y] <= p:
                tp = tp + 1
                ltp.append((x,y))
            elif img_pred[x,y] and distance_gt[x,y] > p:
                fp = fp + 1
                lfp.append((x,y))
            elif (not img_pred[x,y]) and img_gt[x,y] and distance_pred[x,y] > p:
                fn = fn + 1.0
                lfn.append((x,y))
    if (tp+fn != 0):
        COM = tp / (tp+fn)
    else:
        COM = 0
    if (tp+fp != 0):
        COR = tp / (tp+fp)
    else:
        COR = 0
    if ((tp+fp+fn) != 0):    
        Q = tp / (tp+fp+fn)
    else: 
        Q = 0
    if COM+COR == 0:
        F1 = 0
    else:
        F1 = (2*COM*COR) / (COM+COR)
    return tp, fp, fn, COM, COR, Q, F1, ltp, lfp, lfn

def precision_recall_f1(gt_x, gt_y, pred_x, pred_y, filename, max_distance, point_size, w, h, img):

    if len(gt_x) == 0 and len(pred_x) == 0:
        return 1.0, 1.0, 1.0

    n_pred = len(pred_x)
    n_gt = len(gt_x)
    distances_matrix = np.zeros((n_gt, n_pred))

    for p in range(n_pred):
        for g in range(n_gt):
            distance = ((pred_x[p]-gt_x[g])**2 + (pred_y[p]-gt_y[g])**2)**0.5
            if distance > max_distance:
                distances_matrix[g,p] = img.width*img.height*10
            else:
                distances_matrix[g,p] = distance

    row_ind, col_ind = linear_sum_assignment(distances_matrix)

    tp = []
    for t in range(len(row_ind)):
        if distances_matrix[row_ind[t], col_ind[t]] <= max_distance:
            tp.append((row_ind[t], col_ind[t]))

    fn = np.ones((n_gt,))
    fp = np.ones((n_pred,))
    for t in range(len(tp)):
        g, p = tp[t]
        fn[g] = 0
        fp[p] = 0

    fn = np.where(fn)
    fp = np.where(fp)
    fn = fn[0]
    fp = fp[0]

    img_circles = Image.new('RGBA', size = (img.width, img.height), color = (255, 255, 255, 0))
    canvas = ImageDraw.Draw(img_circles)

    for t in range(len(fn)):
        g = fn[t]
        canvas.ellipse((gt_y[g] - max_distance, gt_x[g] - max_distance, gt_y[g] + max_distance, gt_x[g] + max_distance), fill=(255, 0, 0, 50))
        #canvas.ellipse((gt_y[g] - point_size, gt_x[g] - point_size, gt_y[g] + point_size, gt_x[g] + point_size), fill=(255, 0, 0, 255))

    for t in range(len(tp)):
        g, p = tp[t]
        canvas.ellipse((gt_y[g] - max_distance, gt_x[g] - max_distance, gt_y[g] + max_distance, gt_x[g] + max_distance), fill=(255, 255, 0, 50))

    for t in range(len(tp)):
        g, p = tp[t]
        canvas.ellipse((pred_y[p] - point_size, pred_x[p] - point_size, pred_y[p] + point_size, pred_x[p] + point_size), fill=(0, 0, 255, 255))

    for t in range(len(fp)):
        p = fp[t]
        canvas.ellipse((pred_y[p] - point_size, pred_x[p] - point_size, pred_y[p] + point_size, pred_x[p] + point_size), fill=(255, 0, 0, 255))

    assert len(tp)+len(fn) == n_gt
    assert len(tp)+len(fp) == n_pred

    #print('TP: %d\n' % len(tp))
    #print('FP: %d\n' % len(fp))
    #print('FN: %d\n' % len(fn))
    if (len(tp) + len(fp)) > 0:
        precision = float(len(tp)) / (len(tp) + len(fp))
    else:
        precision = 0.
    if (len(tp) + len(fn)) > 0:
        recall = float(len(tp)) / (len(tp) + len(fn))
    else:
        recall = 0.

    img = Image.alpha_composite(img, img_circles)
    img.save(filename)

    ap = precision
    ar = recall
    if (ap+ar) > 0:
        f1 = 2*((ap*ar) / (ap+ar))
    else:
        f1 = 0

    #print('final')
    #print('average precision: %f' % ap)
    #print('average recall: %f' % ar)
    #print('F-measure: %f' % f1)

    return ap, ar, f1

