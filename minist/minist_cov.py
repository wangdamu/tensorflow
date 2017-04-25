import DataLoader
import MnistLoader
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import const

class Network:


    dataloader = None

    # 输入长度
    inputLen = None
    # 输出长度
    outputLen = None


    def __init__(self, dataloader, optimizerType):
        self.dataloader = dataloader
        self.dataloader.load()

        self.xr, self.xc, y0 = self.dataloader.getDataShape()
        self.inputLen = self.xr * self.xc
        self.outputLen = y0

        self.optimizerType = optimizerType

        #img = np.reshape(x0, [28, 28])
        #plt.imshow(img);
        #plt.show()

        pass

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def train(self):
        with tf.name_scope("io"):
            x = tf.placeholder(tf.float32, shape=[None, self.inputLen])
            y_ = tf.placeholder(tf.float32, shape=[None, self.outputLen])

        with tf.name_scope("conv1"):
            W_conv1 = self.weight_variable([5, 5, 1, 32])
            b_conv1 = self.bias_variable([32])

            x_image = tf.reshape(x, [-1,self.xr,self.xc,1])
            h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
            h_pool1 = self.max_pool_2x2(h_conv1)

        with tf.name_scope("conv2"):
            W_conv2 = self.weight_variable([5, 5, 32, 64])
            b_conv2 = self.bias_variable([64])

            h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = self.max_pool_2x2(h_conv2)

        with tf.name_scope("fc1"):
            print(self.xr / 4 * self.xc / 4 * 64)
            W_fc1 = self.weight_variable([int(round(self.xr / 4 * self.xc / 4 * 64)), 1024])
            b_fc1 = self.bias_variable([1024])

            h_pool2_flat = tf.reshape(h_pool2, [-1, int(round(self.xr / 4 * self.xc / 4 * 64))])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        with tf.name_scope("dropout"):
            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        with tf.name_scope("output"):
            W_fc2 = self.weight_variable([1024, 10])
            b_fc2 = self.bias_variable([10])

            y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        with tf.name_scope("cross_entropy"):
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
            if self.optimizerType == const.GradientDescentOptimizer:
                train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
            elif  self.optimizerType == const.AdamOptimizer:
                train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for i in range(20000):
                batch_xs, batch_ys = self.dataloader.getTrainData(50)
                sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
                if i%100 == 0:
                    train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_: batch_ys, keep_prob: 1.0})
                    print("step %d, training accuracy %g"%(i, train_accuracy))
                    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    #test_x, test_y = self.dataloader.getTestData(0)
                    #print("step=%i, accuracy=" % (_), sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))


            test_x, test_y = self.dataloader.getTestData(0)
            print("test accuracy %g"%accuracy.eval(feed_dict={x: test_x, y_:test_y, keep_prob: 1.0}))

        pass


dataloader = MnistLoader.MnistLoader()
network = Network(dataloader=dataloader, optimizerType=const.AdamOptimizer)
network.train()
