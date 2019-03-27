from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
from progress.bar import Bar
import importlib
import numpy as np
import tensorflow as tf
from data import get_data_provider
from ImageNetReading import image_processing
FLAGS = tf.app.flags.FLAGS
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.metrics import confusion_matrix
import itertools
import scipy.misc
import os
CIFAR10_LABELS_LIST = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]



def evaluate(model, dataset,
        batch_size=256,
        checkpoint_dir='./checkpoint'):
    with tf.Graph().as_default() as g:
        print(FLAGS.checkpoint_dir)
        preds = []
        labels = []
        data = get_data_provider(dataset, training=False)
        with tf.device('/cpu:0'):
            if FLAGS.dataset == "imagenet" :
                x, yt =image_processing.inputs(data,batch_size=batch_size,num_preprocess_threads=FLAGS.num_threads)
            else :
                x, yt = data.generate_batches(batch_size)
            is_training = tf.placeholder(tf.bool,[],name='is_training')

        # Build the Graph that computes the logits predictions
        y = model(x, is_training=False)

        # Calculate predictions.
        softmax = tf.nn.softmax(y)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=yt, logits=y))
        accuracy_top1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(y,yt,1), tf.float32))
        accuracy_top5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(y,yt,5), tf.float32))
        # Restore the moving average version of the learned variables for eval.
        #variable_averages = tf.train.ExponentialMovingAverage(
        #    MOVING_AVERAGE_DECAY)
        #variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver()#variabimport matplotlib.pyplot as pltles_to_restore)


        # Configure options for session
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(
                config=tf.ConfigProto(
                            log_device_placement=False,
                            allow_soft_placement=True,
                            gpu_options=gpu_options,
                            )
                        )
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir+'/')
        if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found')
            return

         # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                       start=True))

            num_batches = int(math.ceil(data.size[0] / batch_size))
            total_top1 = 0  # Counts the number of correct predictions per batch.
            total_top5 = 0  # Counts the number of correct predictions per batch.
            total_loss = 0 # Sum the loss of predictions per batch.
            step = 0
            bar = Bar('Evaluating', max=num_batches,suffix='%(percent)d%% eta: %(eta)ds')
            while step < num_batches and not coord.should_stop():
              image, pred_onehot, label, softmax_val, acc_top1,acc_top5,loss_val = sess.run([x, y, yt, softmax,accuracy_top1, accuracy_top5,loss])
              total_top1 += acc_top1
              total_top5 += acc_top5
              total_loss += loss_val
              step += 1
              print(total_top1/step)
              pred = np.argmax(pred_onehot, axis=1)
              preds.extend(pred)
              labels.extend(label)
              for i in range(batch_size):
                  if pred[i] != label[i]:
                      if not os.path.isdir("./" + FLAGS.checkpoint_dir.split('/')[-1] + '_mis'):
                          os.makedirs("./" + FLAGS.checkpoint_dir.split('/')[-1] + '_mis')
                      img = image[i,:,:,:]
                      confidence = np.max(softmax_val[i,:])
                      label_string = CIFAR10_LABELS_LIST[label[i]]
                      mislabel = CIFAR10_LABELS_LIST[pred[i]]
                      scipy.misc.imsave("./" + FLAGS.checkpoint_dir.split('/')[-1] + '_mis/' + label_string + '_' + mislabel + "_" + str(confidence) + '.jpg', img)
              bar.next()

            # Compute precision and loss
            total_top1 /= num_batches
            total_top5 /= num_batches
            total_loss /= num_batches
            conf_mat = confusion_matrix(labels, preds)
            np.set_printoptions(precision=2)
            # plt.figure()
            plot_confusion_matrix(conf_mat, classes=CIFAR10_LABELS_LIST, normalize=True, title='{} - Accuracy: {}%'.format(FLAGS.model_name, np.around(total_top1*100,1)))
            print(conf_mat)
            print(total_top1)
            bar.finish()


        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads)
        return total_top1,total_top5,total_loss

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig = plt.figure()
    plt.tight_layout()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = np.around(cm.astype('float') * 100 / cm.sum(axis=1)[:, np.newaxis],1)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    fig.savefig('./conf_mat/' + FLAGS.checkpoint_dir.split('/')[-1] + '_conf_mat.png', bbox_inches="tight")

def main(argv=None):  # pylint: disable=unused-argument
  m = importlib.import_module('.' + FLAGS.model_name, 'models')
  evaluate(m.model, FLAGS.dataset, checkpoint_dir=FLAGS.checkpoint_dir)


if __name__ == '__main__':
  tf.app.flags.DEFINE_string('checkpoint_dir', './results/model',
                             """Directory where to read model checkpoints.""")
  tf.app.flags.DEFINE_string('dataset', 'cifar10',
                             """Name of dataset used.""")
  tf.app.flags.DEFINE_string('model_name', 'model',
                             """Name of loaded model.""")

  # FLAGS.log_dir = FLAGS.checkpoint_dir+'/log/'
      # Build the summary operation based on the TF collection of Summaries.
      # summary_op = tf.merge_all_summaries()

      # summary_writer = tf.train.SummaryWriter(log_dir)
          # summary = tf.Summary()
          # summary.ParseFromString(sess.run(summary_op))
          # summary.value.add(tag='accuracy/test', simple_value=precision)
          # summary_writer.add_summary(summary, global_step)

  tf.app.run()
