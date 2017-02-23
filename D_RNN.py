from __future__ import print_function
import tensorflow as tf
class d_bi_RNN:
    # ==========
    #   MODEL
    # ==========

    # Parameters
    learning_rate = 0.001
    training_iters = 1000000
    batch_size = 31
    display_step = 10

    # Network Parameters
    input_len = 28
    seq_max_len = 28 # Sequence max length
    n_hidden = 64 # hidden layer num of features
    n_classes = 10 # linear sequence or not

    # Define weights
    weights = {
        'out': tf.Variable(tf.truncated_normal([2*n_hidden, n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.truncated_normal([n_classes]))
    }

    def __init__(self):
        # tf Graph input
        self.x_input = tf.placeholder(shape = [None, self.input_len, self.seq_max_len],dtype= tf.float32)
        self.y_input = tf.placeholder(dtype= tf.float32, shape = [None, self.n_classes])

        self.x_ = tf.transpose(self.x_input, perm= [1,0,2])
        self.x1 = tf.reshape(self.x_, shape= [-1,self.input_len])
        self.x = tf.split(split_dim=0, num_split=self.seq_max_len, value=self.x1)

        # Define a lstm cell with tensorflow
        self.lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden, forget_bias=1.0,
                                                         input_size= [None, self.input_len])
        self.lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden, forget_bias=1.0,
                                                         input_size= [None, self.input_len])

        # Get lstm cell output, providing 'sequence_length' will perform dynamic
        # calculation.
        self.outputs ,_,_= tf.nn.bidirectional_rnn(cell_fw= self.lstm_fw_cell,
                                               cell_bw= self.lstm_bw_cell,
                                               inputs= self.x,
                                               dtype= tf.float32)
                                               # sequence_length= self.seq_max_len)

        # When performing dynamic calculation, we must retrieve the last
        # dynamically computed output, i.e., if a sequence length is 10, we need
        # to retrieve the 10th output.
        # However TensorFlow doesn't support advanced indexing yet, so we build
        # a custom op that for each sample in batch size, get its length and
        # get the corresponding relevant output.

        # 'outputs' is a list of output at every timestep, we pack them in a Tensor
        # and change back dimension to [batch_size, n_step, n_input]

        # self.outputs_ = tf.pack(self.outputs[0])
        # self.outputs_ = tf.reshape(tf.concat(1, self.outputs_), [-1, self.n_hidden])

        # outputs = tf.pack(outputs[0])
        # outputs = tf.transpose(outputs, [1, 0, 2])
        #
        # # Hack to build the indexing and retrieve the right output.
        # batch_size = tf.shape(outputs)[0]
        # # Start indices for each sample
        # index = tf.range(0, batch_size) * self.seq_max_len + (seqlen - 1)
        # # Indexing
        # outputs = tf.gather(tf.reshape(outputs, [-1, self.n_hidden]), index)
        # self.op_tensor = tf.pack(self.outputs)
        # self.op = tf.transpose(a= self.op_tensor, perm=[1, 0, 2])
        # Linear activation, using outputs computed above
        self.pred = tf.matmul(self.outputs[-1], self.weights['out']) + self.biases['out']

        # Define loss and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.pred, self.y_input))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # Evaluate model
        self.correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y_input, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))


    # def dynamicRNN(self, x, seqlen, weights, biases):
    #
    #     # Prepare data shape to match `rnn` function requirements
    #     # Current data input shape: (batch_size, n_steps, n_input)
    #     # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    #
    #     # Permuting batch_size and n_steps
    #     x = tf.transpose(x, [1, 0, 2])
    #     # Reshaping to (n_steps*batch_size, n_input)
    #     x = tf.reshape(x, [-1, 1])
    #     # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    #     x = tf.split(0, self.seq_max_len, x)
    #
    #     # Define a lstm cell with tensorflow
    #     lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden, forget_bias= 1.0)
    #     lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden, forget_bias= 1.0)
    #
    #     # Get lstm cell output, providing 'sequence_length' will perform dynamic
    #     # calculation.
    #     outputs= tf.nn.bidirectional_rnn(cell_fw=lstm_fw_cell, cell_bw=lstm_bw_cell,inputs=x, dtype=tf.float32,
    #                                 sequence_length=seqlen)
    #
    #     # When performing dynamic calculation, we must retrieve the last
    #     # dynamically computed output, i.e., if a sequence length is 10, we need
    #     # to retrieve the 10th output.
    #     # However TensorFlow doesn't support advanced indexing yet, so we build
    #     # a custom op that for each sample in batch size, get its length and
    #     # get the corresponding relevant output.
    #
    #     # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    #     # and change back dimension to [batch_size, n_step, n_input]
    #
    #     outputs = tf.pack(outputs[0])
    #     outputs = tf.reshape(tf.concat(1,outputs), [-1, self.n_hidden])
    #
    #     # outputs = tf.pack(outputs[0])
    #     # outputs = tf.transpose(outputs, [1, 0, 2])
    #     #
    #     # # Hack to build the indexing and retrieve the right output.
    #     # batch_size = tf.shape(outputs)[0]
    #     # # Start indices for each sample
    #     # index = tf.range(0, batch_size) * self.seq_max_len + (seqlen - 1)
    #     # # Indexing
    #     # outputs = tf.gather(tf.reshape(outputs, [-1, self.n_hidden]), index)
    #
    #     # Linear activation, using outputs computed above
    #     return tf.matmul(outputs, weights['out']) + biases['out']