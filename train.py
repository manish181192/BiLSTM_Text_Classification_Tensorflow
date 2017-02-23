import tensorflow as tf
from D_RNN import d_bi_RNN
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# Import MNIST data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
model_path = "mnist_DLSTM_classifier/models/model_13_02_1/model.ckpt"
rnn = d_bi_RNN()
# Initializing the variables
init = tf.global_variables_initializer()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    dev_max_accuracy = 0.9
    dev_min_loss = 50
    model_saver = tf.train.Saver()
    # Keep training until reach max iterations
    while step * rnn.batch_size < rnn.training_iters:

        batch_x, batch_y = mnist.train.next_batch(rnn.batch_size)
        batch_x = np.reshape(batch_x, [rnn.batch_size, rnn.input_len, rnn.seq_max_len])

        dev_x = mnist.validation.images[:rnn.batch_size]
        dev_y = mnist.validation.labels[:rnn.batch_size]
        dev_x = np.reshape(dev_x, [-1, rnn.input_len, rnn.seq_max_len])

        # Run optimization op (backprop)
        _, tr_acc, tr_loss = sess.run([rnn.optimizer, rnn.accuracy, rnn.cost],
                                feed_dict= {rnn.x_input: np.array(batch_x), rnn.y_input: np.array(batch_y)})

        if step % rnn.display_step == 0:
            # Calculate batch accuracy and loss on dev set
            dev_acc, dev_loss = sess.run([rnn.accuracy, rnn.cost],
                     feed_dict={rnn.x_input: np.array(dev_x), rnn.y_input: np.array(dev_y)})

            print(" TRAIN DATA")
            print("Iter " + str(step*rnn.batch_size) + ", Minibatch Loss= " +
                  "{:.6f}".format(tr_loss) + ", Training Accuracy= " +
                  "{:.5f}".format(tr_acc))
            print(" \nDEV DATA\n")

            if dev_acc>dev_max_accuracy:
                dev_max_accuracy = dev_acc
                save_path =  model_saver.save(sess= sess,save_path= model_path)
                print("Model saved in", save_path)

            if dev_loss<dev_min_loss:
                dev_min_loss = dev_loss
            print("Loss", dev_loss)
            print("accuracy", dev_acc)
            print("Min Loss", dev_min_loss)
            print("Max accuracy", dev_max_accuracy)
        step += 1
    print("Optimization Finished!")
    test_data = mnist.test.images[:rnn.batch_size]
    test_label = mnist.test.labels[:rnn.batch_size]
    test_data = np.reshape(test_data, newshape= [rnn.batch_size, rnn.input_len, rnn.seq_max_len])
    model_saver = tf.train.Saver()
    model_saver.restore(sess=sess, save_path=model_path)
    print ("Session Restored")

    print("Testing Accuracy:",
          sess.run(rnn.accuracy, feed_dict={rnn.x_input: test_data, rnn.y_input: test_label}))
