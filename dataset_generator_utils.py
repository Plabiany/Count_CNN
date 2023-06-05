import numpy as np
from scipy import ndimage
import cv2
from skimage.morphology import skeletonize

def rect(x):
    return where(abs(x)>=0.9, 1, 0)

def get_heatmap(x, y, width, height, sigma, output_width=None, output_height=None, square = None):
  if output_width == None:
    output_width = width
    output_height = height
  fx = output_width / float(width)
  fy = output_height / float(height)
  grid_x = np.arange(output_width)
  grid_y = np.arange(output_height)
  double_sigma2 = 2 * sigma * sigma
  heatmap = np.zeros((output_width, output_height), dtype=np.float)

  for i in range(len(x)):
    px = (x[i]*fx)
    py = (y[i]*fy)
    if square:
      heatmap[int(py)-1:int(py)+1 , int(px)-1:int(px)+1] = 1
    else:
      exp_x = np.exp(-(grid_x-px)**2/double_sigma2)
      exp_y = np.exp(-(grid_y-py)**2/double_sigma2)
      exp = np.outer(exp_y, exp_x)
      heatmap = np.maximum(heatmap, exp)
  return heatmap, None

def get_edt(x, y, width, height, output_width, output_height):
  points = np.ones((output_width, output_height), dtype=np.float)

  if len(x) == 0:
    return points

  fx = output_width / float(width)
  fy = output_height / float(height)
  for i in range(len(x)):
    px = int(x[i]*fx)
    py = int(y[i]*fy)
    py = min(py, output_width-1)
    px = min(px, output_height-1)
    points[py,px] = 0

  edt = ndimage.distance_transform_edt(points)
  print(np.max(edt))
  edt = edt / np.max(edt)
  return edt

def get_voronoi(x, y, width, height, output_width, output_height):
  # Read in the image.
  img = np.zeros((output_width, output_height), dtype=np.uint8)
  if len(x) <= 1:
    return img
   
  # Rectangle to be used with Subdiv2D
  rect = (0, 0, output_width, output_height)
    
  # Create an instance of Subdiv2D
  subdiv = cv2.Subdiv2D(rect)

  # Create an array of points.
  fx = output_width / float(width)
  fy = output_height / float(height)
  for i in range(len(x)):
    px = int(x[i]*fx)
    py = int(y[i]*fy)
    py = min(py, output_width-1)
    px = min(px, output_height-1)
    subdiv.insert((px,py))

  # Allocate space for Voronoi Diagram
  img_voronoi = np.zeros(img.shape, dtype = np.uint8)

  # Draw Voronoi diagram
  draw_voronoi(img_voronoi,subdiv)
  #cv2.imwrite('temp/voronoi_antes.png', (img_voronoi).astype(np.uint8))
  img_voronoi = skeletonize(img_voronoi > 0)
  #cv2.imwrite('temp/voronoi.png', (img_voronoi*255).astype(np.uint8))
  # Show results
  return img_voronoi

# Draw voronoi diagram
def draw_voronoi(img, subdiv) :
  ( facets, centers) = subdiv.getVoronoiFacetList([])
  for i in range(0,len(facets)) :
      ifacet_arr = []
      for f in facets[i] :
          ifacet_arr.append(f)
      
      ifacet = np.array(ifacet_arr, np.int)
      color = 0

      cv2.fillConvexPoly(img, ifacet, color, cv2.LINE_AA, 0);
      ifacets = np.array([ifacet])
      cv2.polylines(img, ifacets, True, 255, 1, cv2.LINE_AA, 0)
      #cv2.circle(img, (centers[i][0], centers[i][1]), 3, (0, 0, 0), cv2.CV_FILLED, cv2.LINE_AA, 0)