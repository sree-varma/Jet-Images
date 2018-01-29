"""
The machine learning part is included in this code the deep CNN is created to classify the images
(https://arxiv.org/pdf/1612.01551.pdf) section 3.2.

"""

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
#from sklearn.metrics import confusion_matrix
#import sklearn.metrics
from sklearn.metrics import roc_curve, auc
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


def conv_layer(input, num_input_channels, filter_size, num_filters,drp,name="conv"):
    with tf.name_scope(name):
        shape = [filter_size, filter_size, num_input_channels, num_filters]
        w = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="B")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        act=tf.nn.dropout(conv,drp)
        act = tf.nn.relu(act + b)
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
img_size = 33

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

conv1 = conv_layer(input=x_image, num_input_channels=num_channels, filter_size=filter_size1, num_filters=num_filters1,drp=0.5,name="conv1")
conv2 = conv_layer(input=conv1,num_input_channels=num_filters1,filter_size=filter_size2, num_filters=num_filters2,drp=0.25,name="conv2")
conv3 = conv_layer(input=conv2,num_input_channels=num_filters2,filter_size=filter_size3, num_filters=num_filters3,drp=0.25,name="conv3")


layer_flat, num_features = flatten_layer(conv3)

fc1 = fc_layer(input=layer_flat,num_inputs=num_features,num_outputs=fc_size, name="fc1")
activated_fc1 = tf.nn.relu(fc1)
fc2 = fc_layer(input=activated_fc1, num_inputs=fc_size, num_outputs=num_classes, name="fc2")
#y_pred = fc2
#y_pred_cls = tf.argmax(y_pred, dimension=1)
y_prob = tf.nn.softmax(fc_2)
y_pred = tf.argmax(y_prob, dimension=1)

"""
Learning configuration
"""

# batch size
batch_size = 128
# validation split
validation_size = .2
learning_rate = 0.00005

# how long to wait after validation loss stops improving before terminating training
early_stopping = None # use None if you don't want to implement early stopping


"""
Data set Configuration
"""

train_path ='/usr/mixed_images/50h-50p/train/'
test_path ='/usr/Herwig/colour/images_500-550GeV/test/'
test_path1='/usr/Pythia/colour/images_500-550GeV/test/'

data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)
test_images, test_ids,test_labels = dataset.read_test_set(test_path, img_size,classes)
test_pythia_images,test_pythia_ids,test_pythia_labels=dataset.read_test_set(test_path1,img_size,classes)
print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(test_images)))
print("- Validation-set:\t{}".format(len(data.valid.labels)))

num_batches = int(data.train.num_examples/batch_size)

"""
Loss Function + Optimization
"""

with tf.name_scope("cost"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc2, labels=y_true)
    cost = tf.reduce_mean(cross_entropy)
    tf.summary.scalar("cost", cost)
with tf.name_scope("Optimize"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

session.run(tf.global_variables_initializer())

logdir = 'logs/classifier/colour/feature_pythia500'
writer = tf.summary.FileWriter(logdir)
writer.add_graph(session.graph)

eval_writer = tf.summary.FileWriter(logdir + '_eval')  
x_batch, _, _, _ = data.train.next_batch(batch_size)

# Add image summary to inspect network input
tf.summary.image('input', x_batch)
merged_summary = tf.summary.merge_all() 
saver = tf.train.Saver()
result = []
valid=[]
classifier=[]
labels=[]
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
        # validation step
        validate_acc, val_loss,y_result = session.run([accuracy, cost,y_pred], feed_dict=feed_dict_validate)
        av_validate_acc = av_validate_acc + validate_acc
        av_val_loss = av_val_loss + val_loss
	
        if i % num_batches == 0:            
            epoch = int(i / num_batches)
            av_val_loss /= num_batches
            av_train_acc /= num_batches
            av_validate_acc /= num_batches


            summary_value = session.run(merged_summary,feed_dict_train)
            writer.add_summary(summary_value, i)

            summary_value_val = session.run(merged_summary, feed_dict_validate)
            eval_writer.add_summary(summary_value_val, i)
	    classifier_features=session.run([activated_fc1],feed_dict=feed_dict_train)
	    classifier.append(classifier_features)
	    #labels.append(y_true_batch)
            msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}," \
                  " Training Loss: {3:.3f}, Validation Loss: {4:.3f}"
            print(msg.format(epoch + 1, av_train_acc, av_validate_acc, train_loss,  av_val_loss))
            av_train_acc = 0.0
            av_validate_acc = 0.0
            av_val_loss = 0.0
            

	final_iteration = starting_iteration + num_iterations
	#print y_result, y_valid_batch
#	if int(i % num_batches) >= ((num_iterations % num_batches)-1):
	#result.append(y_result)
	#valid.append(y_valid_batch)
 #   y_score=np.concatenate(result, axis=0 )
#    print y_score.shape
#    y_valid=np.concatenate(valid,axis=0)
#    print y_valid.shape


    features_class=np.concatenate(classifier,axis=0)
    
    #feature_labels=np.concatenate(labels,axis=0)
    
    features_class=np.concatenate(features_class,axis=1)
    #print features_class.shape
    features_classifier=pd.DataFrame(features_class)
    features_classifier.to_csv("Features_Pythia_500.csv",index=False)
    
	
#    fprs, tprs = [None] * 2, [None] * 2
#    aucs = [None] * 2
#    for i in range(2):
#    	fprs[i], tprs[i], _ = roc_curve(y_valid[:, i], y_score[:, i])
#    	aucs[i] = auc(fprs[i], tprs[i], reorder=True)

    #plt.figure(figsize=(8, 8))
    #plt.plot([0, 1], [0, 1], '--', color='black')

    #plt.title('One-vs-rest ROC curves', fontsize=16)

    #plt.xlabel('Quark Jet Efficiency')
    #plt.ylabel('Gluon Jet Rejection')
    #g_r=(tprs[0]+fprs[1])
    #q_e=(tprs[1]+fprs[0])
    #for i in range(2):
    #plt.plot(1-fprs[0],tprs[0], label='%s (AUC %.2lf)' % (classes[0], aucs[0]))
    
    #plt.ylim((0.5,3.5))
    #plt.xlim((0.0,1.0))
    #plt.plot(tprs[0],(tprs[0]/np.sqrt(fprs[0])))
    
    #fprs_quarks_pythia= pd.DataFrame({"fprs_quarks_pythia":fprs[0]})
    #tprs_quarks_pythia= pd.DataFrame({"tprs_quarks_pythia":tprs[0]})
    #fprs_quarks_pythia.to_csv("Herwig_fprs_pyth_500_valid_color.csv",index=False)
    #tprs_quarks_pythia.to_csv("Herwig_tprs_pyth_500_valid_color.csv",index=False)
     
    #print tprs[0]
    #print fprs[0]
    #plt.legend(fontsize=14)
    #plt.plot(q_e,g_r)
    #plt.show()
    
    return final_iteration

optimize(num_iterations=70310)#35156)#2812)#7031)




def sample_prediction(test_im,test_label):
    
    feed_dict_test = {x: test_im.reshape(1, img_size, img_size, num_channels),y_true:np.array([[1, 0]])}#np.array([[1, 0]])}

    test_pred = session.run(y_pred, feed_dict=feed_dict_test)
    return test_pred
output=[]
output_pythia=[]
fprs, tprs = [None] * 2, [None] * 2
aucs = [None] * 2

fprs_p, tprs_p = [None] * 2, [None] * 2
aucs_p = [None] * 2

for i in range(0,len(test_images)):
	output.append(sample_prediction(test_images[i],test_labels))
for i in range(0,len(test_pythia_images)):
	output_pythia.append(sample_prediction(test_pythia_images[i],test_pythia_labels))
	

y_test=np.concatenate(output, axis=0 )
#y_test=y_test.astype(np.float64)

y_test_pythia=np.concatenate(output_pythia, axis=0 )

#y_test_herwig=y_test_herwig.astype(np.float64)
#print y_test
#test_labels=test_labels.astype(np.float64)
#test_herwig_labels=test_herwig_labels.astype(np.float64)
#print y_test
for i in range(2):
    	fprs[i], tprs[i], _ = roc_curve((test_labels[:, i]),y_test[:,i])
	fprs_p[i],tprs_p[i],_=roc_curve(test_pythia_labels[:, i],y_test_pythia[:,i])
	aucs[i] = auc(fprs[i], tprs[i], reorder=True)
	aucs_p[i] = auc(fprs_p[i], tprs_p[i], reorder=True)
print fprs[0]
print tprs[0]

fprs_quarks_herwig_df = pd.DataFrame({"fprs_quarks_herwig":fprs[0]})
tprs_quarks_herwig_df= pd.DataFrame({"tprs_quarks_herwig":tprs[0]})
fprs_quarks_pythia_df = pd.DataFrame({"fprs_quarks_pythia":fprs_p[0]})
tprs_quarks_pythia_df= pd.DataFrame({"tprs_quarks_pythia":tprs_p[0]})



fprs_quarks_herwig_df.to_csv("Mixed_fprs_her_500_color_50h-50p.csv",index=False)
tprs_quarks_herwig_df.to_csv("Mixed_tprs_her_500_color_50h-50p.csv",index=False)
fprs_quarks_pythia_df.to_csv("Mixed_fprs_pyth_500_color_50h-50p.csv",index=False)
tprs_quarks_pythia_df.to_csv("Mixed_tprs_pyth_500_color_50h-50p.csv",index=False)



