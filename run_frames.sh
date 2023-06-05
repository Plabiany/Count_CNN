echo '1/1'
python /content/drive/MyDrive/Count2021/train_frames.py \
-width 512 \
-height 512 \
-num_epochs 100 \
-batch_size 4 \
-sigma 1.1 \
-min_sigma 1.1 \
-stages 4 \
-n_frames 2 \
-layer_name block3_conv4 \
-model vgg19 \
-experiment_folder /content/drive/MyDrive/Count2021/DatasetPintado_OlhoCalda \
-train_images_folder /content/drive/MyDrive/Count2021/DatasetPintado_OlhoCalda/TRAIN \
-val_images_folder /content/drive/MyDrive/Count2021/DatasetPintado_OlhoCalda/TRAIN
