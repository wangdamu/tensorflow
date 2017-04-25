from tensorflow.examples.tutorials.mnist import input_data
import DataLoader

class MnistLoader(DataLoader.DataLoader):
    '''
    Mnist数据加载器
    '''

    mnist = None

    def __init__(self):
        super().__init__()

    def load(self):
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    def getTrainData(self, count):
        batch_xs, batch_ys = self.mnist.train.next_batch(count)
        return batch_xs, batch_ys

    def getTestData(self, count):
        return self.mnist.test.images, self.mnist.test.labels

    def getDataShape(self):
        return 28, 28, 10
