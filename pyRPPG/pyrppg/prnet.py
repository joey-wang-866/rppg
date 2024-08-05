import numpy as np
from tensorflow.contrib.framework import arg_scope
import tensorflow.contrib.layers as tcl
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # hide warning


def residual_block(input_tensor, num_outputs, kernel_size=4, stride=1,
                   activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, scope=None):
    """
    Residual block for a neural network.

    :param input_tensor: Input tensor for the residual block
    :type input_tensor: Tensor
    :param num_outputs: Number of output channels
    :type num_outputs: int
    :param kernel_size: Kernel size for the convolutional layers (default: 4).
    :type kernel_size: int, optional
    :param stride: Stride for the convolutional layers (default: 1).
    :type stride: int, optional
    :param activation_fn: Activation function for the layers (default: tensorflow.nn.relu).
    :type stride: function, optional
    :param normalizer_fn: Normalization function for the layers (default: tensorflow.contrib.layers.batch_norm).
    :type normalizer_fn: function, optional
    :param scope: Scope for the variable names (default: None).
    :type scope: str, optional
    :return: Output tensor of the residual block
    :rtype: Tensor
    """
    assert num_outputs % 2 == 0  # num_outputs must be divisible by channel_factor (2 here)
    with tf.variable_scope(scope, 'residual_block'):
        shortcut = input_tensor
        if stride != 1 or input_tensor.get_shape()[3] != num_outputs:
            shortcut = tcl.conv2d(shortcut, num_outputs, kernel_size=1, stride=stride,
                                  activation_fn=None, normalizer_fn=None, scope='shortcut')
        x = tcl.conv2d(input_tensor, num_outputs // 2,
                       kernel_size=1, stride=1, padding='SAME')
        x = tcl.conv2d(x, num_outputs // 2, kernel_size=kernel_size,
                       stride=stride, padding='SAME')
        x = tcl.conv2d(x, num_outputs, kernel_size=1, stride=1,
                       activation_fn=None, padding='SAME', normalizer_fn=None)
        x += shortcut
        x = normalizer_fn(x)
        x = activation_fn(x)
    return x


class ResFCN256:
    """
    ResFCN256 class that represents the neural network architecture.

    :param resolution_inp: Input resolution (default: 256).
    :type resolution_inp: int, optional
    :param resolution_op: Output resolution (default: 256).
    :type resolution_op: int, optional
    :param channel: Number of channels (default: 3).
    :type resolution_op: int, optional
    :param name: Name of the neural network architecture (default: resfcn256).
    :type name: str, optional
    """

    def __init__(self, resolution_inp=256, resolution_op=256, channel=3, name='resfcn256'):
        self.name = name
        self.channel = channel
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op

    def __call__(self, input_tensor, is_training=True):
        """
        Construct the ResFCN256 network architecture.

        :param input_tensor: The input tensor to the network
        :type input_tensor: Tensor
        :param is_training: Indicates whether the network is in training mode (default: True).
        :type is_training: bool, optional
        :return: Output tensor after processing through the network
        :rtype: Tensor
        """
        with tf.variable_scope(self.name) as scope:
            with arg_scope([tcl.batch_norm], is_training=is_training, scale=True):
                with arg_scope([tcl.conv2d, tcl.conv2d_transpose], activation_fn=tf.nn.relu,
                               normalizer_fn=tcl.batch_norm, biases_initializer=None, padding='SAME',
                               weights_regularizer=tcl.l2_regularizer(0.0002)):
                    size = 16
                    # input_tensor: s x s x 3
                    se = tcl.conv2d(input_tensor, num_outputs=size,
                                    kernel_size=4, stride=1)  # 256 x 256 x 16
                    se = residual_block(
                        se, num_outputs=size * 2, kernel_size=4, stride=2)  # 128 x 128 x 32
                    se = residual_block(
                        se, num_outputs=size * 2, kernel_size=4, stride=1)  # 128 x 128 x 32
                    se = residual_block(
                        se, num_outputs=size * 4, kernel_size=4, stride=2)  # 64 x 64 x 64
                    se = residual_block(
                        se, num_outputs=size * 4, kernel_size=4, stride=1)  # 64 x 64 x 64
                    se = residual_block(
                        se, num_outputs=size * 8, kernel_size=4, stride=2)  # 32 x 32 x 128
                    se = residual_block(
                        se, num_outputs=size * 8, kernel_size=4, stride=1)  # 32 x 32 x 128
                    se = residual_block(
                        se, num_outputs=size * 16, kernel_size=4, stride=2)  # 16 x 16 x 256
                    se = residual_block(
                        se, num_outputs=size * 16, kernel_size=4, stride=1)  # 16 x 16 x 256
                    se = residual_block(
                        se, num_outputs=size * 32, kernel_size=4, stride=2)  # 8 x 8 x 512
                    se = residual_block(
                        se, num_outputs=size * 32, kernel_size=4, stride=1)  # 8 x 8 x 512

                    pd = tcl.conv2d_transpose(
                        se, size * 32, 4, stride=1)  # 8 x 8 x 512
                    pd = tcl.conv2d_transpose(
                        pd, size * 16, 4, stride=2)  # 16 x 16 x 256
                    pd = tcl.conv2d_transpose(
                        pd, size * 16, 4, stride=1)  # 16 x 16 x 256
                    pd = tcl.conv2d_transpose(
                        pd, size * 16, 4, stride=1)  # 16 x 16 x 256
                    pd = tcl.conv2d_transpose(
                        pd, size * 8, 4, stride=2)  # 32 x 32 x 128
                    pd = tcl.conv2d_transpose(
                        pd, size * 8, 4, stride=1)  # 32 x 32 x 128
                    pd = tcl.conv2d_transpose(
                        pd, size * 8, 4, stride=1)  # 32 x 32 x 128
                    pd = tcl.conv2d_transpose(
                        pd, size * 4, 4, stride=2)  # 64 x 64 x 64
                    pd = tcl.conv2d_transpose(
                        pd, size * 4, 4, stride=1)  # 64 x 64 x 64
                    pd = tcl.conv2d_transpose(
                        pd, size * 4, 4, stride=1)  # 64 x 64 x 64

                    pd = tcl.conv2d_transpose(
                        pd, size * 2, 4, stride=2)  # 128 x 128 x 32
                    pd = tcl.conv2d_transpose(
                        pd, size * 2, 4, stride=1)  # 128 x 128 x 32
                    pd = tcl.conv2d_transpose(
                        pd, size, 4, stride=2)  # 256 x 256 x 16
                    pd = tcl.conv2d_transpose(
                        pd, size, 4, stride=1)  # 256 x 256 x 16

                    pd = tcl.conv2d_transpose(
                        pd, 3, 4, stride=1)  # 256 x 256 x 3
                    pd = tcl.conv2d_transpose(
                        pd, 3, 4, stride=1)  # 256 x 256 x 3
                    # , padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
                    pos = tcl.conv2d_transpose(
                        pd, 3, 4, stride=1, activation_fn=tf.nn.sigmoid)

                    return pos

    @property
    def vars(self):
        """
        Get the variables associated with the ResFCN256 network.

        :return: A list of variables associated with the network
        :rtype: List[tf.Variable]
        """
        return [var for var in tf.global_variables() if self.name in var.name]


class PosPrediction:
    """
    PosPrediction class for predicting facial landmarks.
    """

    def __init__(self, resolution_inp=256, resolution_op=256):
        """
        Constructor.

        :param resolution_inp: Input resolution (default: 256).
        :type resolution_inp: int, optional
        :param resolution_op: Output resolution (default: 256).
        :type resolution_op: int, optional
        """
        # Hyper settings
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op
        self.max_pos = resolution_inp * 1.1
        # Network type
        self.network = ResFCN256(self.resolution_inp, self.resolution_op)
        # Net forward
        self.x = tf.placeholder(
            tf.float32, shape=[None, self.resolution_inp, self.resolution_inp, 3])
        self.x_op = self.network(self.x, is_training=False)
        self.sess = tf.Session(config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True)))

    def restore(self, model_path):
        """
        Restores the pre-trained model from the given path.

        :param model_path: Path to the pre-trained model
        :type model_path: str
        """
        tf.train.Saver(self.network.vars).restore(self.sess, model_path)

    def predict(self, image):  # RGB
        """
        Predicts facial landmarks for a single input image.

        :param image: Input image in RGB format
        :type image: numpy.ndarray
        :return: Predicted facial landmarks
        :rtype: numpy.ndarray
        """
        pos = self.sess.run(self.x_op, feed_dict={
                            self.x: image[np.newaxis, :, :, :]})
        pos = np.squeeze(pos)
        return pos * self.max_pos  # scaled to 256 * 1.1

    def predict_batch(self, images):
        """
        Predicts facial landmarks for a batch of input images.

        :param images: Input images in RGB format
        :type images: numpy.ndarray
        :return: Predicted facial landmarks for each image
        :rtype: numpy.ndarray
        """
        pos = self.sess.run(self.x_op, feed_dict={self.x: images})
        return pos * self.max_pos  # scaled to 256 * 1.1
