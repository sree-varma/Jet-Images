import tensorflow as tf

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


def fc_layer(input, num_inputs, num_outputs, activation_fn, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_outputs]), name="B")
        pre_act = tf.matmul(input, w) + b
        act = activation_fn(pre_act)
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


def inference(image, num_classes):
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
    fc_size = 128  # Number of neurons in fully-connected layer.

    # Number of color channels for the images: 1 channel for gray-scale.
    num_channels = 3

    conv1 = conv_layer(input=image, num_input_channels=num_channels,
                       filter_size=filter_size1, num_filters=num_filters1, name="conv1")
    conv2 = conv_layer(input=conv1, num_input_channels=num_filters1,
                       filter_size=filter_size2, num_filters=num_filters2, name="conv2")
    conv3 = conv_layer(input=conv2, num_input_channels=num_filters2,
                       filter_size=filter_size3, num_filters=num_filters3, name="conv3")

    layer_flat, num_features = flatten_layer(conv3)

    fc1 = fc_layer(input=layer_flat, num_inputs=num_features, num_outputs=fc_size, activation_fn=tf.nn.relu, name="fc1")

    logits = fc_layer(input=fc1, num_inputs=fc_size, num_outputs=num_classes, activation_fn=tf.identity, name="fc2")

    return logits, fc1
