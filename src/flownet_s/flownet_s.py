from ..net import Net, Mode
from ..utils import LeakyReLU, average_endpoint_error, pad, antipad
from ..downsample import downsample
import math
import tensorflow as tf
slim = tf.contrib.slim


class FlowNetS(Net):
    def __init__(self, mode=Mode.TRAIN, debug=False, reuse=None):
        super(FlowNetS, self).__init__(mode=mode, debug=debug)
        self.reuse = reuse

    def model(self, inputs, training_schedule, trainable=True):
        """ Returns Encoder output of FlowNetS CNN """
        _, height, width, _ = inputs['input_a'].shape.as_list()
        stacked = False
        with tf.variable_scope('FlowNetS', reuse=self.reuse):
            concat_inputs = tf.concat([inputs['input_a'], inputs['input_b']], axis=3)
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                # Only backprop this network if trainable
                                trainable=trainable,
                                # He (aka MSRA) weight initialization
                                weights_initializer=slim.variance_scaling_initializer(),
                                activation_fn=LeakyReLU,
                                # We will do our own padding to match the original Caffe code
                                padding='VALID'):

                weights_regularizer = slim.l2_regularizer(training_schedule['weight_decay'])
                with slim.arg_scope([slim.conv2d], weights_regularizer=weights_regularizer):
                    with slim.arg_scope([slim.conv2d], stride=2):
                        conv_1 = slim.conv2d(pad(concat_inputs, 3), 64, 7, scope='conv1')
                        conv_2 = slim.conv2d(pad(conv_1, 2), 128, 5, scope='conv2')
                        conv_3 = slim.conv2d(pad(conv_2, 2), 256, 5, scope='conv3')

                    conv3_1 = slim.conv2d(pad(conv_3), 256, 3, scope='conv3_1')
                    with slim.arg_scope([slim.conv2d], num_outputs=512, kernel_size=3):
                        conv4 = slim.conv2d(pad(conv3_1), stride=2, scope='conv4')
                        conv4_1 = slim.conv2d(pad(conv4), scope='conv4_1')
                        conv5 = slim.conv2d(pad(conv4_1), stride=2, scope='conv5')
                        conv5_1 = slim.conv2d(pad(conv5), scope='conv5_1')
                    conv6 = slim.conv2d(pad(conv5_1), 1024, 3, stride=2, scope='conv6')
                    conv6_1 = slim.conv2d(pad(conv6), 1024, 3, scope='conv6_1')
                    print conv6_1
                    return conv6_1

    def loss(self, flow, predictions):
        flow = flow * 0.05

        losses = []
        INPUT_HEIGHT, INPUT_WIDTH = float(flow.shape[1].value), float(flow.shape[2].value)

        # L2 loss between predict_flow6, blob23 (weighted w/ 0.32)
        predict_flow6 = predictions['predict_flow6']
        size = [predict_flow6.shape[1], predict_flow6.shape[2]]
        downsampled_flow6 = downsample(flow, size)
        losses.append(average_endpoint_error(downsampled_flow6, predict_flow6))

        # L2 loss between predict_flow5, blob28 (weighted w/ 0.08)
        predict_flow5 = predictions['predict_flow5']
        size = [predict_flow5.shape[1], predict_flow5.shape[2]]
        downsampled_flow5 = downsample(flow, size)
        losses.append(average_endpoint_error(downsampled_flow5, predict_flow5))

        # L2 loss between predict_flow4, blob33 (weighted w/ 0.02)
        predict_flow4 = predictions['predict_flow4']
        size = [predict_flow4.shape[1], predict_flow4.shape[2]]
        downsampled_flow4 = downsample(flow, size)
        losses.append(average_endpoint_error(downsampled_flow4, predict_flow4))

        # L2 loss between predict_flow3, blob38 (weighted w/ 0.01)
        predict_flow3 = predictions['predict_flow3']
        size = [predict_flow3.shape[1], predict_flow3.shape[2]]
        downsampled_flow3 = downsample(flow, size)
        losses.append(average_endpoint_error(downsampled_flow3, predict_flow3))

        # L2 loss between predict_flow2, blob43 (weighted w/ 0.005)
        predict_flow2 = predictions['predict_flow2']
        size = [predict_flow2.shape[1], predict_flow2.shape[2]]
        downsampled_flow2 = downsample(flow, size)
        losses.append(average_endpoint_error(downsampled_flow2, predict_flow2))

        loss = tf.losses.compute_weighted_loss(losses, [0.32, 0.08, 0.02, 0.01, 0.005])

        # Return the 'total' loss: loss fns + regularization terms defined in the model
        return tf.losses.get_total_loss()
