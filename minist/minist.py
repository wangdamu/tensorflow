import DataLoader
import MnistLoader
import tensorflow as tf

class Network:

    dataloader = None

    # 输入长度
    inputLen = None
    # 输出长度
    outputLen = None

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.dataloader.load()

        x0, y0 = self.dataloader.getTrainData(1);
        _, self.inputLen = x0.shape
        _, self.outputLen = y0.shape

        pass

    def train(self):
        with tf.name_scope("input"):
            x = tf.placeholder(tf.float32, [None, self.inputLen])
            W = tf.Variable(tf.random_normal(shape=[self.inputLen, self.outputLen], seed=1234))
            b = tf.Variable(tf.random_normal(shape=[self.outputLen], seed=5678))

        with tf.name_scope("output"):
            y = tf.nn.softmax(tf.matmul(x, W) + b)
            y_ = tf.placeholder(tf.float32, [None, self.outputLen])

        with tf.name_scope("cross_entropy"):
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
            train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for _ in range(1000):
                batch_xs, batch_ys = self.dataloader.getTrainData(100)
                sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
                correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
                if(_ % 100 == 0):
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    test_x, test_y = self.dataloader.getTestData(0)
                    print("step=%i, accuracy=" % (_), sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))

            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            test_x, test_y = self.dataloader.getTestData(0)
            print(sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))

        pass


dataloader = MnistLoader.MnistLoader()
network = Network(dataloader=dataloader)
network.train()
