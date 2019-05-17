

import tensorflow as tf
import skimage.io as io
import numpy

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
output_dim = {'image_dim': [32, 32, 3]}

tfrecords_filename = 'validation.tfrecords'

def read_and_decode(filename_queue):
    
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'channel': tf.FixedLenFeature([], tf.int64),
        'image_hr': tf.FixedLenFeature([], tf.string),
        'image_lr': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape

    image_hr = tf.decode_raw(features['image_hr'],  out_type=tf.uint8)
    image_lr = tf.decode_raw(features['image_lr'],  out_type=tf.uint8)

    label = tf.cast(features['label'], tf.int32)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    channel = tf.cast(features['channel'], tf.int32)

    OUTPUT_DIM = output_dim['image_dim']

    image_hr = tf.reshape(image_hr, OUTPUT_DIM)
    image_lr = tf.reshape(image_lr, OUTPUT_DIM)
    
    # images, annotations = tf.train.shuffle_batch( [resized_image, resized_annotation],
    #                                              batch_size=2,
    #                                              capacity=30,
    #                                              num_threads=2,
    #                                              min_after_dequeue=10)

    # image_hr = tf.train.shuffle_batch( [image_hr],
    #                                     batch_size=1,
    #                                     capacity=30,
    #                                     num_threads=2,
    #                                     min_after_dequeue=10)
    
    return image_hr, image_lr




######################

filename_queue = tf.train.string_input_producer(
    [tfrecords_filename], num_epochs=10)

# Even when reading in multiple threads, share the filename
# queue.

image_hr, image_lr = read_and_decode(filename_queue)

# The op for initializing the variables.
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

with tf.Session()  as sess:
    
    sess.run(init_op)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    # Let's read off 3 batches just for example
    for i in range(3):
    
        image_hr, image_lr = sess.run([image_hr, image_lr])
        print('current batch')
        
        # We selected the batch size of two
        # So we should get two image pairs in each batch
        # Let's make sure it is random

        io.imshow(image_hr)
        io.show()

        io.imshow(image_lr)
        io.show()
        

    coord.request_stop()
    coord.join(threads)
