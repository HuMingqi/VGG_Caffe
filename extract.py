import numpy as np
import caffe
import os
import json

model='/mnt/g/machine_learning/models/vgg/caffe/VGG_ILSVRC_19_layers_deploy.prototxt';
weights='/mnt/g/machine_learning/models/vgg/caffe/snapshots/clothes_vgg_iter_45500.caffemodel';
imageset='/mnt/g/machine_learning/dataset/clothes_amazon'
output='/mnt/g/machine_learning/models/vgg/caffe/output/feature_lib'
BGR_mean= np.array([104,117,123])	# ilsvrc_2012_channel_mean_pixel

def extract():
	caffe.set_mode_cpu()
	# caffe.set_mode_gpu()
	net=caffe.Net(model, weights, caffe.TEST);

	# create transformer for the input called 'data', resize
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
	transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
	transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
	transformer.set_mean('data', BGR_mean)         # subtract the dataset-mean value in each channel
	
	net.blobs['data'].reshape(1,        # batch size
	                          3,         # 3-channel (BGR) images
	                          224, 224)  # image size is 224*224
	
	fc6_list=[]; fc7_list=[]; fc8_list=[]
	id_list=[]
	for sub in os.listdir(imageset):
		subp= imageset+'/'+sub+'/src'
		for img in os.listdir(subp):
				# return an image with type np.float32 in range [0, 1]
			image = caffe.io.load_image(subp+'/'+img)
			transformed_image = transformer.preprocess('data', image)
				# copy a littler ndarray to another ndarray is valid, auto-filling
			net.blobs['data'].data[...]= transformed_image
				#*** use deploy.prototxt; if not, call forward like "net.forward(start='conv1', end='fc8')"
			net.forward()
			
			fc6=net.blobs['fc6'].data[0]	# blob name not layer name, output ndarray
			fc7=net.blobs['fc7'].data[0]
			fc8=net.blobs['fc8'].data[0]
				# normalize
			fc6_norm= l2_normlize(fc6)
			fc7_norm= l2_normlize(fc7)
			fc8_norm= l2_normlize(fc8)
				# print(fc8, fc8_norm)

			fc6_list.append(fc6_norm.tolist())
			fc7_list.append(fc7_norm.tolist())
			fc8_list.append(fc8_norm.tolist())
			id_list.append(subp+'/'+img)
			print(img+' done')

	save(id_list, fc6_list, output+'/vgg_fc6_features.json')
	save(id_list, fc7_list, output+'/vgg_fc7_features.json')
	save(id_list, fc8_list, output+'/vgg_fc8_features.json')
	print('Extract done')


def l2_normlize(vec):
	l2_norm= np.linalg.norm(vec, 2)
	if l2_norm!=0:
		return vec/l2_norm
	else:
		return vec

def save(id_list, feature_list, file_nm):
    '''
    	Save as json obj like {id:feature, ...}, A feature is a vector.
		Overwrite file if re-call.
    '''
    # Note! dict will auto-update duplicate key, so don't warry about extraction for the same image.
    dic = dict(list(zip(id_list, feature_list)))
    # mode='w' not 'a', in python2, no encoding param!
    with open(file_nm, 'w') as file:
    	# Note! dic does not keep sequence,
    	# Use 'sort_keys'.(Of course it's does not matter if not use, just looks very chaotic)
    	file.write(json.dumps(dic, sort_keys=True))


if __name__ == '__main__':
	extract()