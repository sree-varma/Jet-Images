"""
The machine learning part is included in this code the deep CNN is created to classify the images
(https://arxiv.org/pdf/1612.01551.pdf) section 3.2.

"""

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import random

import dataset

#gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.1)
#s = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

"""
Helper Functions
"""


def conv_layer(input, num_input_channels, filter_size, num_filters, name="conv"):
    with tf.name_scope(name):
        shape = [filter_size, filter_size, num_input_channels, num_filters]
        w = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="B")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        act = tf.nn.relu(conv + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
    return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def fc_layer(input, num_inputs, num_outputs, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_outputs]), name="B")
        act = tf.matmul(input, w) + b
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
    return act


def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features

"""
Network Definition
"""
convolutional = True

# Convolutional Layer 1.filters
filter_size1 = 8  
num_filters1 = 64

# Convolutional Layer 2.
filter_size2 = 4
num_filters2 = 64

# Convolutional Layer 3.
filter_size3 = 4
num_filters3 = 64
    
# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.

# Number of color channels for the images: 1 channel for gray-scale.
num_channels = 3

# image dimensions (only squares for now)
img_size = 32

# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# class info

classes = ['quarks', 'gluons']
num_classes = len(classes)


tf.reset_default_graph()
session = tf.Session()

if not convolutional:
    x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
else:
    x = tf.placeholder(tf.float32, shape=[None, img_shape[0], img_shape[1], num_channels], name='x')


x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
tf.summary.image('input', x_image, 3)
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

conv1 = conv_layer(input=x_image, num_input_channels=num_channels, filter_size=filter_size1, num_filters=num_filters1, name="conv1")
#conv_out = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
conv2 = conv_layer(input=conv1,num_input_channels=num_filters1,filter_size=filter_size2, num_filters=num_filters2,name="conv2")
conv3 = conv_layer(input=conv2,num_input_channels=num_filters2,filter_size=filter_size3, num_filters=num_filters3,name="conv3")

layer_flat, num_features = flatten_layer(conv3)

fc1 = fc_layer(input=layer_flat,num_inputs=num_features,num_outputs=fc_size, name="fc1")
sigmoid = tf.nn.sigmoid(fc1)
tf.summary.histogram("fc1/sigmoid", sigmoid)
fc2 = fc_layer(input=sigmoid,num_inputs=fc_size,num_outputs=num_classes, name="fc2")

y_pred = fc2
y_pred_cls = tf.argmax(y_pred, dimension=1)


"""
Learning configuration
"""

# batch size
batch_size = 25
# validation split
validation_size = .2
learning_rate = 0.00005

# how long to wait after validation loss stops improving before terminating training
early_stopping = None  # use None if you don't want to implement early stopping


"""
Data set Configuration
"""

train_path = 'data/jet_images_sample/train'
test_path = 'data/jet_images_sample/test'

data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)
test_images, test_ids, labels = dataset.read_test_set(test_path, img_size, classes)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(test_images)))
print("- Validation-set:\t{}".format(len(data.valid.labels)))

num_batches = int(data.train.num_examples/batch_size)

"""
Loss Function + Optimization
"""

with tf.name_scope("cost"):
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=fc2, labels=y_true)
    cost = tf.reduce_mean(cross_entropy)
    tf.summary.scalar("cost", cost)
with tf.name_scope("Optimize"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

session.run(tf.global_variables_initializer())

logdir = 'logs/classifier/10'
writer = tf.summary.FileWriter(logdir)
writer.add_graph(session.graph)

eval_writer = tf.summary.FileWriter(logdir + '_eval')

x_batch, _, _, _ = data.train.next_batch(batch_size)

# Add image summary to inspect network input
tf.summary.image('input', x_batch)
merged_summary = tf.summary.merge_all()


def optimize(num_iterations, starting_iteration=0):

    av_train_acc = 0.0
    av_validate_acc = 0.0
    av_val_loss = 0.0
    
    for i in range(starting_iteration,
                   starting_iteration + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)

        if not convolutional:
            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, flattened image shape]
            x_batch = x_batch.reshape(batch_size, img_size_flat)
            x_valid_batch = x_valid_batch.reshape(batch_size, img_size_flat)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.

        feed_dict_train = {x: x_batch, y_true: y_true_batch}
        feed_dict_validate = {x: x_valid_batch, y_true: y_valid_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.

        # training step
        _, train_acc, train_loss = session.run([optimizer, accuracy, cost], feed_dict=feed_dict_train)
        av_train_acc = av_train_acc + train_acc

        # saver = tf.train.Saver()
        # saver.save(session, 'my_test_model')

        # validation step
        validate_acc, val_loss = session.run([accuracy, cost], feed_dict=feed_dict_validate)
        av_validate_acc = av_validate_acc + validate_acc
        av_val_loss = av_val_loss + val_loss

        if i % num_batches == 0:

            summary_value = session.run(merged_summary, feed_dict_train)
            writer.add_summary(summary_value, i)

            summary_value_val = session.run(merged_summary, feed_dict_validate)
            eval_writer.add_summary(summary_value_val, i)
            
            epoch = int(i / num_batches)
            av_val_loss /= num_batches
            av_train_acc /= num_batches
            av_validate_acc /= num_batches

            msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}," \
                  " Training Loss: {3:.3f}, Validation Loss: {4:.3f}"
            print(msg.format(epoch + 1, av_train_acc, av_validate_acc, train_loss,  av_val_loss))
            av_train_acc = 0.0
            av_validate_acc = 0.0
            av_val_loss = 0.0

    final_iteration = starting_iteration + num_iterations
    return final_iteration


optimize(num_iterations=10000)
