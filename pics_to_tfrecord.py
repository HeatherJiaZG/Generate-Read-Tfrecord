#coding:utf-8
import sys
import os
import cv2
import numpy as np
import struct
import tensorflow as tf 
from PIL import Image

height, width = 288, 288

def _get_output_filename(output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    return '%s/train_2.tfrecord' % (output_dir)

def _process_image_withoutcoder(filename):

	image = Image.open(filename)
	image = np.array(image)
	assert len(image.shape) == 3
	# height = image.shape[0]
	# width = image.shape[1]
	channel = image.shape[2]

	return image, channel

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    """Wrapper for insert float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _convert_to_example_simple(image, channel):

	new_h, new_w = height, width
	h,w,_ = image.shape
	top = np.random.randint(0, h - new_h)
	left = np.random.randint(0, w - new_w)
	image_hr = image[top: top + new_h,
			left: left + new_w]
	image_lr = cv2.resize(image_hr, dsize=(height//4, width//4))
	image_lr = cv2.resize(image_lr, dsize=(height, width), interpolation=cv2.INTER_CUBIC)

	# from matplotlib import pyplot as plt
	# plt.imshow(image_hr)
	# plt.show()
	# plt.imshow(image_lr)
	# plt.show()

	# print("lr_shape="+str(image_lr.shape))
	# print("hr_shape="+str(image_hr.shape))

	example = tf.train.Example(features=tf.train.Features(feature={
        'image_hr': _bytes_feature(tf.compat.as_bytes(image_hr.tostring())),
	    'image_lr': _bytes_feature(tf.compat.as_bytes(image_lr.tostring())),
	    # 'height': _int64_feature(height),
	    # 'width': _int64_feature(width),
	    # 'channel': _int64_feature(channel),
        'age': _int64_feature(0)
    }))
	return example


def _add_to_tfrecord(PicName, tfrecord_writer):

	# image_data, height, width, channel = _process_image_withoutcoder(PicName)
	image_data, channel = _process_image_withoutcoder(PicName)
	# example = _convert_to_example_simple(image_data, height, width, channel )
	example = _convert_to_example_simple(image_data, channel )
	tfrecord_writer.write(example.SerializeToString())



def readName(pic_dir, out_dir):

	tf_filename = _get_output_filename(out_dir)
	# print("tf_filename"+ tf_filename)
	g = os.walk(pic_dir)
	# print(len(g))
	idx = 0

	print ('<<<<<<<<<<<<<<<  START CONVERT  >>>>>>>>>>>>>>>>>>')
	
	with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
		for path, dir, filelist in g:
			# print("path="+str(path)+",dir="+str(dir)+",filelist="+str(filelist))
			for filename in filelist:
				# print ("filename"+filename)
				if filename.endswith(".png"):
					# print("filename="+filename)
					idx += 1	
					PicName = os.path.join(path, filename)
					_add_to_tfrecord(PicName, tfrecord_writer)



if __name__ == '__main__':
	# pic_dir =  "F:\\GLOW_code\\DIV2K\\train250" #"../../../srgan/srgan/data2017/DIV2K_train_HR"
	pic_dir = "../Brain_img/imgs/DIV2K_train_HR"
	out_dir = "../Brain_img/2D/train"
	readName(pic_dir, out_dir)

