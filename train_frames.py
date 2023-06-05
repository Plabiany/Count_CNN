import argparse
import os
import sys

import tensorflow as tf
#import keras

from glob import glob
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model
from keras.utils import plot_model
from model_frames import get_training_model
from dataset_generator_frames2 import ImageHeatmapGenerator

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def get_last_epoch_and_model_file(WEIGHT_DIR, epoch=None):

	WEIGHTS_SAVE = 'model_{epoch:06d}.h5'
	if not os.path.isdir(WEIGHT_DIR):
	    os.makedirs(WEIGHT_DIR)

	if epoch is not None and epoch != '': #override
	    return int(epoch),  WEIGHT_DIR + '/' + WEIGHTS_SAVE.format(epoch=epoch)

	files = [file for file in glob(WEIGHT_DIR + '/model_*.h5')]
	files = [file.split('/')[-1] for file in files]
	epochs = [file.split('_')[1] for file in files]
	epochs = [file.split('.')[0] for file in epochs]
	epochs = [int(epoch) for epoch in epochs if epoch.isdigit() ]
	if 'model_final.h5' in files:
	        return -1, os.path.join(WEIGHT_DIR, 'model_final.h5')
	elif len(epochs) == 0:
		return None, None
	else:
	    ep = max([int(epoch) for epoch in epochs])
	    return ep, os.path.join(WEIGHT_DIR, WEIGHTS_SAVE.format(epoch=ep))


parser = argparse.ArgumentParser(description='Training Counting CNN')
parser.add_argument('-width', action="store", required=True, help='image width', dest='img_w', type=int)
parser.add_argument('-height', action="store", required=True, help='image height', dest='img_h', type=int)
parser.add_argument('-num_epochs', action="store", required=True, help='number of epochs', dest='nb_epoch', type=int)
parser.add_argument('-n_frames', action="store", required=True, help='number of frames', dest='n_frames', type=int)
parser.add_argument('-stages', action="store", required=True, help='number of stages', dest='stages', type=int)
parser.add_argument('-batch_size', action="store", required=True, help='batch size', dest='batch_size', type=int)
parser.add_argument('-sigma', action="store", required=True, help='sigma', dest='sigma', type=float)
parser.add_argument('-min_sigma', action="store", required=True, help='sigma', dest='min_sigma', type=float)
parser.add_argument('-experiment_folder', action="store", required=True, help='folder to save the experiment', dest='experiment_folder')
parser.add_argument('-train_images_folder', action="store", required=True, help='training rgb images folder', dest='train_path_imgs')
#parser.add_argument('-train_labels_folder', action="store", required=True, help='training ground-truth images folder', dest='train_path_ann')
parser.add_argument('-val_images_folder', action="store", required=True, help='validation rgb images folder', dest='validation_path_imgs')
#parser.add_argument('-val_labels_folder', action="store", required=True, help='validation ground-truth images folder', dest='validation_path_ann')
parser.add_argument('-layer_name', action="store", required=True, help='layer name of the CNN, such as block4_conv4 for VGG19, conv2d_75 for inceptionresnet)', dest='layer_name')
parser.add_argument('-model', action="store", required=True, help='Model (vgg19, inceptionresnet)', dest='model_name')
#parser.add_argument('-continue', action="store_true", default=False, required=False, help='continue training from last epoch', dest='continue_train')
parser.add_argument('-lr', action="store", default=0.01, required=False, help='learning rate', dest='learning_rate', type=float)
parser.add_argument('-momentum', action="store", default=0.9, required=False, help='momentum', dest='momentum', type=float)
parser.add_argument('-upsampling', action="store", default=False, required=False, help='upsampling', dest='upsampling', type=bool)

arguments = parser.parse_args()
experiment_folder = arguments.experiment_folder 
nb_epoch = arguments.nb_epoch
batch_size = arguments.batch_size
train_path_imgs = arguments.train_path_imgs
#train_path_ann = arguments.train_path_ann
validation_path_imgs = arguments.validation_path_imgs
#validation_path_ann = arguments.validation_path_ann
#model_name = arguments.model_name
continue_train = True
learning_rate = arguments.learning_rate
momentum = arguments.momentum
img_w = arguments.img_w
img_h = arguments.img_h
sigma = arguments.sigma
min_sigma = arguments.min_sigma
stages = arguments.stages
layer_name = arguments.layer_name
model_name = arguments.model_name
n_frames = arguments.n_frames
upsampling = arguments.upsampling
temporal = True
psp = False
weights = 'imagenet'
square = False

#experiment_name = experiment_folder +'/count_sigmax_'+ str(sigma).replace('.','_') + '_sigmin_' + str(min_sigma).replace('.','_') + '_frames_' + str(n_frames)
configs = 'count_sigmax_' + str(sigma).replace('.','_') + '_sigmin_' + str(min_sigma).replace('.','_') + '_frames_' + str(n_frames) + '_stages_' + str(stages) + '_temporal_' + str(temporal)
experiment_name = os.path.join(experiment_folder, configs)
#print 'UPSAMPLING: ' + str(upsampling)

initial_epoch, model_path = get_last_epoch_and_model_file(experiment_name)

if initial_epoch == -1:
	print('Treinamento finalizado.')
	sys.exit()
elif initial_epoch == None or continue_train == False:
	model, _ = get_training_model(img_w=img_w, img_h=img_h, img_d=3, n_frames=n_frames, model_name=model_name, layer_name=layer_name, weights='imagenet', stages = stages, upsampling=upsampling, psp=psp, np_branch1 = 2 if temporal else 0, np_branch2 = 1, np_branch3 = 0)
	#model.summary()
	initial_epoch = 0
else:
	print(model_path)
	#model = get_training_model(img_w=img_w, img_h=img_h, img_d=3, n_frames=n_frames, model_name=model_name, layer_name=layer_name, weights='imagenet', np_branch1 = 2, np_branch2 = 1, stages=stages, upsampling=upsampling, psp=psp)
	model = tf.keras.models.load_model(model_path)
	#model = load_model(model_path)
	#model = model.load_weights(model_path)

output_width = output_height = model.output[0].get_shape()[1]
print('\n\n\nDimensao de saida: (%d,%d)\n\n\n' % (output_width, output_height))

train_imageHeatmapGenerator = ImageHeatmapGenerator(train_path_imgs)
train_generator = train_imageHeatmapGenerator.generator(batch_size, sigma, stages, n_frames, img_w, img_h, temporal, output_width=output_width, output_height=output_height, min_sigma=min_sigma, is_sequential=False, square = square)
#steps_per_epoch = train_imageHeatmapGenerator.n_images // batch_size
steps_per_epoch = 100

val_imageHeatmapGenerator = ImageHeatmapGenerator(validation_path_imgs)
val_generator = val_imageHeatmapGenerator.generator(batch_size, sigma, stages, n_frames, img_w, img_h, temporal, output_width=output_width, output_height=output_height, min_sigma=min_sigma, is_sequential=True, square = square)
validation_steps = val_imageHeatmapGenerator.n_images // batch_size
#####

optimizer = SGD(lr=learning_rate, momentum=momentum, decay=0.0005, nesterov=False)
#model.compile(loss='mse', 'mse', 'mse', 'mse', 'mse', 'mse', 'categorical_crossentropy'], optimizer=optimizer, metrics=['mse', 'mse', 'mse', 'mse', 'mse', 'mse', 'acc'])
model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

filepath = os.path.join(experiment_name, "model_{epoch:06d}.h5")
checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=False, save_freq='epoch', save_weights_only=False)
tensorboard = TensorBoard(os.path.join(experiment_name,'logs'), write_graph=False)
callbacks_list = [checkpoint, tensorboard]

history = model.fit_generator(train_generator, callbacks=callbacks_list, steps_per_epoch=steps_per_epoch,
		epochs=nb_epoch, initial_epoch=initial_epoch, verbose=1) #,validation_data=val_generator, validation_steps=validation_steps)

model.save(os.path.join(experiment_name, "model_final.h5"))

#model.save_weights(os.path.join(experiment_name, "model_weights_final.h5"))