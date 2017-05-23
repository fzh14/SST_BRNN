import tensorflow as tf
import gensim
import string
import time
import os

##### prepare data
path = 'stanfordSentimentTreebank/train_input.txt'
dev_path = 'stanfordSentimentTreebank/dev_input.txt'
test_path = 'stanfordSentimentTreebank/test_input.txt'
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
model = gensim.models.KeyedVectors.load_word2vec_format(project_path + '/glove.6B/glove.6B.50d.txt', binary=False)

sentence_max = 56


class Data(object):
    def __init__(self, path):
        self.data_in = []
        self.data_label = []
        self.batch_id = 0
        self.data_length = []

        fp = open(path, 'r')
        for l in fp.readlines():
            line = l.strip('\n').split('|')
            word_list = line[0].split(' ')
            s = []
            for item in word_list:
                item = string.lower(item)
                s.append(model[item].tolist())
            if len(word_list) < sentence_max:
                for i in range(sentence_max - len(word_list)):
                    s.append([0. for k in range(50)])
            self.data_length.append(len(word_list))
            l = [0. for k in range(5)]
            value = float(line[1])
            label_index = int(value / 0.2)
            if label_index >= 5:
                l[4] = 1.0
            else:
                l[label_index] = 1.0
            self.data_in.append(s)
            self.data_label.append(l)

    def next(self, batch_size):
        if self.batch_id + batch_size >= len(self.data_in):
            batch_data_in = self.data_in[self.batch_id: len(self.data_in)]
            batch_data_label = self.data_label[self.batch_id: len(self.data_in)]
            batch_data_length = self.data_length[self.batch_id: len(self.data_in)]
            self.batch_id = self.batch_id + batch_size - len(self.data_in)
            batch_data_in += self.data_in[0:self.batch_id]
            batch_data_label += self.data_label[0:self.batch_id]
            batch_data_length += self.data_length[0:self.batch_id]
        else:
            batch_data_in = self.data_in[self.batch_id: self.batch_id + batch_size]
            batch_data_label = self.data_label[self.batch_id: self.batch_id + batch_size]
            batch_data_length = self.data_length[self.batch_id: self.batch_id + batch_size]
            self.batch_id = self.batch_id + batch_size
        return batch_data_in, batch_data_label, batch_data_length


trainset = Data(path)
testset = Data(test_path)
devset = Data(dev_path)

print len(trainset.data_in)

# ==============
#     MODEL
# ==============

learning_rate = 0.001
training_iters = 1000000
batch_size = 128
display_step = 500

# Network Parameters
n_input = 50  # data input (shape: 50*56)
n_steps = 56  # timesteps
n_hidden = 128  # hidden layer num of features
n_classes = 5  # total classes

x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
z = tf.placeholder(tf.int32, [None])

weights = {
    # (50, 128)
    'in': tf.Variable(tf.random_normal([n_input, n_hidden])),
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    # (2*128, 5)
    'out': tf.Variable(tf.random_normal([2 * n_hidden, n_classes]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden, ])),
    'out': tf.Variable(tf.random_normal([n_classes, ]))
}


def last_relevant(output, length):
    b_size = tf.shape(output)[0]
    max_length = sentence_max
    out_size = int(output.get_shape()[2])
    #l = length - [1 for k in range(batch_size)]
    index = tf.range(0, b_size) * max_length + (length -1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant


def BiRNN(x, z, weights, biases):
    X = tf.reshape(x, [-1, n_input])
    X_in = tf.matmul(X, weights['in']) + biases['in']
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden])

    # X_in = tf.unstack(X_in, None, 1)

    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    init_fw_state = lstm_fw_cell.zero_state(batch_size, dtype=tf.float32)
    init_bw_state = lstm_bw_cell.zero_state(batch_size, dtype=tf.float32)

    outputs, final_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, X_in, sequence_length=z,
                                                           initial_state_fw=init_fw_state,
                                                           initial_state_bw=init_bw_state,
                                                           time_major=False)

    # outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, X_in,
    #                                              dtype=tf.float32)
    outputs = tf.concat(outputs, 2)
    #outputs = tf.unstack(outputs, n_steps, 1)
    last = last_relevant(outputs, z)
    return tf.matmul(last, weights['out']) + biases['out']


pred = BiRNN(x, z, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

t_acc = 0
dev_x, dev_y, dev_length = trainset.next(batch_size)
f_log = open('logs/log_brnn.txt','w')
f_log.write(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + '\n')

def test():
    t_step = 1
    t_acc = 1.0
    log_str = ''
    while t_step <= 100:
        '''test_x = testset.data_in
        test_y = testset.data_label
        test_length = testset.data_length'''
        test_x, test_y, test_length = testset.next(batch_size)
        acc = sess.run(accuracy, feed_dict={x: test_x, y: test_y, z: test_length})
        t_acc = (acc + t_acc * (t_step - 1)) / (float(t_step))
        log_str = "TEST Accuracy= " + \
                  "{:.5f}".format(t_acc)
        t_step += 1
    print log_str
    f_log.write(log_str + '\n')

def dev():
    d_step = 1
    d_acc = 1.0
    log_str = ''
    while d_step <= 100:
        dev_x, dev_y, dev_length = devset.next(batch_size)
        acc = sess.run(accuracy, feed_dict={x: dev_x, y: dev_y, z: dev_length})
        d_acc = (acc + d_acc * (d_step - 1)) / (float(d_step))
        log_str = "DEV Accuracy= " + \
                  "{:.5f}".format(t_acc)
        d_step += 1
    print log_str

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    step = 1
    while step * batch_size <= training_iters:
        batch_x, batch_y, batch_length = trainset.next(batch_size)
        sess.run(optimizer, feed_dict={
            x: batch_x,
            y: batch_y,
            z: batch_length
        })
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, z: batch_length})
        t_acc = (acc + t_acc*(step -1))/(float(step))
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: dev_x, y: dev_y, z: dev_length})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, z: batch_length})
            log_str = ("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(t_acc) + ",dev Accuracy= " + \
                  "{:.5f}".format(acc))
            print log_str
            f_log.write(log_str + '\n')
            dev()
        step += 1

    print 'Optimizer Complete'
    test()
