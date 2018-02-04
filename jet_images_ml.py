"""
The machine learning part is included in this code the deep CNN is created to classify the images
(https://arxiv.org/pdf/1612.01551.pdf) section 3.2.

"""
import os
import tensorflow as tf

from network import inference
import dataset

#gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.1)
#s = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

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
Learning configuration
"""

# batch size
batch_size = 25
# validation split
validation_size = .2
learning_rate = 0.0005

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
Evaluation Configureation
"""

valdiation_batches = 20
validation_freq = num_batches  # validate once per epoch for now (change to suit data)
train_summary_freq = 1
saving_freq = num_batches  # save once per epoch for now (change to suit data)

"""
Loss Function + Optimization
"""

with tf.name_scope("cost"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
    cost = tf.reduce_mean(cross_entropy)
    tf.summary.scalar("cost", cost)
with tf.name_scope("Optimize"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(y_pred, y_true_cls)
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


saver = tf.train.Saver()


def optimize(num_iterations, starting_iteration=0):

    av_train_acc = 0.0

    for i in range(starting_iteration,
                   starting_iteration + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)

        if not convolutional:
            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, flattened image shape]
            x_batch = x_batch.reshape(batch_size, img_size_flat)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.

        feed_dict_train = {x: x_batch, y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.

        # training step
        _, train_acc, train_loss = session.run([optimizer, accuracy, cost], feed_dict=feed_dict_train)
        av_train_acc = av_train_acc + train_acc

        if i % train_summary_freq == 0:
            summary_value = session.run(merged_summary, feed_dict_train)
            writer.add_summary(summary_value, i)

        if i % validation_freq == 0:
            # validation step
            av_validate_acc = 0.0
            av_val_loss = 0.0
            for j in range(valdiation_batches):
                x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)
                if not convolutional:
                    x_valid_batch = x_valid_batch.reshape(batch_size, img_size_flat)

                feed_dict_validate = {x: x_valid_batch, y_true: y_valid_batch}
                validate_acc, val_loss = session.run([accuracy, cost], feed_dict=feed_dict_validate)
                av_validate_acc = av_validate_acc + validate_acc
                av_val_loss = av_val_loss + val_loss
            av_val_loss /= valdiation_batches
            av_validate_acc /= valdiation_batches
            summary_value_val = session.run(merged_summary, feed_dict_validate)
            eval_writer.add_summary(summary_value_val, i)

        if i % saving_freq:
            if not os.path.exists('saved_models'):
                os.makedirs('save_models')
            saver.save(session, './saved_models/saved_model_{}.ckpt'.format(i))

        if i % num_batches == 0:
            epoch = int(i / num_batches)
            av_train_acc /= num_batches

            msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}," \
                  " Training Loss: {3:.3f}, Validation Loss: {4:.3f}"
            print(msg.format(epoch + 1, av_train_acc, av_validate_acc, train_loss,  av_val_loss))
            av_train_acc = 0.0

    final_iteration = starting_iteration + num_iterations
    return final_iteration


optimize(num_iterations=10000)
