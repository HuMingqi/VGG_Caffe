#!/usr/bin/env sh
$CAFFE_HOME/build/tools/caffe train \
-solver=$ml/models/vgg/caffe/solver.prototxt \
-weights=$ml/models/vgg/caffe/VGG_ILSVRC_19_layers.caffemodel \
2>&1 | tee train.log

#$CAFFE_HOME/build/tools/caffe time -model $ml/models/shoes_classification/train_val_net.prototxt$