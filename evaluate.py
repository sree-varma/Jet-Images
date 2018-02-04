import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve
import matplotlib.pyplot as plt

from network import inference
import dataset


tf.reset_default_graph()
session = tf.Session()


"""
Inputs
"""
convolutional = True

# image dimensions (only squares for now)
img_size = 32

num_channels = 3
# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# class info

classes = ['quarks', 'gluons']
num_classes = len(classes)

if not convolutional:
    x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
else:
    x = tf.placeholder(tf.float32, shape=[None, img_shape[0], img_shape[1], num_channels], name='x')

x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
tf.summary.image('input', x_image, 3)
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)


"""
Inference (Forward Pass)
"""

logits, features = inference(x_image, num_classes=num_classes)

y_probs = tf.nn.softmax(logits)
tf.summary.histogram('probs', y_probs)

y_pred = tf.argmax(logits, dimension=1)


"""
Restore variables
"""

saver = tf.train.Saver()
saver.restore(session,  tf.train.latest_checkpoint('./saved_models'))

"""
Data set Configuration
"""

test_path = 'data/jet_images_sample/test'
test_images, test_ids, labels = dataset.read_test_set(test_path, img_size, classes)

batch_size = 5

print("Size of:")
print("- Test-set:\t\t{}".format(len(test_images)))

num_batches = int(len(test_images)/batch_size)

predictions = []
true_labels = []
for i in range(num_batches):
    test_image_batch = test_images[i * batch_size:(i+1) * batch_size]
    test_label_batch = labels[i * batch_size:(i+1) * batch_size]

    feed_dict = {x: test_image_batch, y_true:  test_label_batch}

    y_pred_value, features_value = session.run([y_pred, features], feed_dict=feed_dict)

    #TODO: save features for reuse later
    predictions.extend(y_pred_value)
    true_labels.extend(list(np.argmax(test_label_batch, axis=1)))

print("Accuracy: {}".format(accuracy_score(true_labels, predictions)))
print("Precision_score: {}".format(precision_score(true_labels, predictions)))
print("Recall_score: {}".format(recall_score(true_labels, predictions)))
print("F1_score: {}".format(f1_score(true_labels, predictions)))
fpr, tpr, thresholds = roc_curve(true_labels, predictions)

plt.figure()
plt.plot(fpr, tpr)
plt.show()
