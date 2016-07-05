import tensorflow as tf

from cifar10_model_pre import _variable_with_weight_decay
from cifar10_model_pre import _variable_on_cpu
from cifar10_model_pre import _activation_summary
import cifar10_input
from inception.slim import ops, scopes

FLAGS = tf.app.flags.FLAGS
NUM_CLASSES = cifar10_input.NUM_CLASSES

def inference(images):
  """Build the CIFAR-10 model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  with scopes.arg_scope([ops.conv2d, ops.fc], stddev=1e-4, bias=0.1, batch_norm_params={}):
      with scopes.arg_scope([ops.conv2d], kernel_size=[5,5], padding='SAME'):
          with scopes.arg_scope([ops.max_pool], kernel_size=[3,3], padding='SAME'):
            net = ops.conv2d(images, num_filters_out=64)
            net = ops.max_pool(net)
            net = ops.conv2d(net, num_filters_out=64)
            net = ops.max_pool(net)
            net = ops.flatten(net)
            net = ops.fc(net, num_units_out=384)
            net = ops.fc(net, num_units_out=192)
            return net
