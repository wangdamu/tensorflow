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

        xr, xc, y0 = self.dataloader.getDataShape()
        self.inputLen = xr * xc
        self.outputLen = y0
        self.optimizerType = optimizerType

        #img = np.reshape(x0, [28, 28])
        #plt.imshow(img);
        #plt.show()

        pass

    def train(self):
        with tf.name_scope("input"):
            x = tf.placeholder(tf.float32, [None, self.inputLen])
            W = tf.Variable(tf.zeros(shape=[self.inputLen, self.outputLen]))
            b = tf.Variable(tf.zeros(shape=[self.outputLen]))

        with tf.name_scope("hidden"):
            hx = tf.matmul(x, W) + b
            hW = tf.Variable(tf.random_normal(shape=[self.inputLen, self.outputLen], seed=1234))
            hb = tf.Variable(tf.random_normal(shape=[self.outputLen], seed=5678))

        with tf.name_scope("output"):
            y = tf.nn.softmax(tf.matmul(x, W) + b)
            y_ = tf.placeholder(tf.float32, [None, self.outputLen])

        with tf.name_scope("cross_entropy"):
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
            if self.optimizerType == const.GradientDescentOptimizer:
                train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
            elif  self.optimizerType == const.AdamOptimizer:
                train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for _ in range(1000):
                batch_xs, batch_ys = self.dataloader.getTrainData(100)
                sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
                correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
                if(_ % 100 == 0):
                    pass
                    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    #test_x, test_y = self.dataloader.getTestData(0)
                    #print("step=%i, accuracy=" % (_), sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))

            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            test_x, test_y = self.dataloader.getTestData(0)
            print(sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))

        pass


dataloader = MnistLoader.MnistLoader()
network = Network(dataloader=dataloader, optimizerType=const.AdamOptimizer)
network.train()
