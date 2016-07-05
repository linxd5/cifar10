
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import cifar10

import os

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():

    global_step = tf.Variable(0, trainable=False)
    images, labels = cifar10.distorted_inputs()
    logits = cifar10.inference(images)
    loss = cifar10.loss(logits, labels)
    train_op = cifar10.train(loss, global_step)
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    saver = tf.train.Saver(tf.all_variables())
    init = tf.initialize_all_variables()
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    true_count = 0
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      _, loss_value, precisions  = sess.run([train_op, loss, top_k_op])
      duration = time.time() - start_time
      true_count += np.sum(precisions)

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        format_str = ('%s: step %d, loss = %.2f, acc = %.3f')
        print (format_str % (duration, step, loss_value, true_count/(FLAGS.batch_size*10)))
        true_count = 0
        

      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
