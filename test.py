import tensorflow as tf
# from train import model_path, mnist
from D_RNN import d_bi_RNN
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
model_path = "mnist_DLSTM_classifier/models/model_13_02_1/model.ckpt"
# Import MNIST data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
rnn = d_bi_RNN()
init = tf.global_variables_initializer()
# Calculate accuracy

with tf.Session() as session:
    session.run(init)
    test_data = mnist.test.images[:rnn.batch_size]
    test_label = mnist.test.labels[:rnn.batch_size]
    test_data = np.reshape(test_data, newshape=[rnn.batch_size, rnn.input_len, rnn.seq_max_len])
    model_saver = tf.train.Saver()
    model_saver.restore(sess=session, save_path=model_path)
    print ("Session Restored")

    print("Testing Accuracy:",
          session.run(rnn.accuracy, feed_dict={rnn.x_input: test_data, rnn.y_input: test_label}))

