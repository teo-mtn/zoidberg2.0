from ctypes import sizeof
from keras.datasets import mnist;
from matplotlib import pyplot;
import numpy as np;
import pandas as pd;
import seaborn as sns;

class Bootstrap:
    train_X = []
    train_Y = []
    test_X = []
    test_Y = []

    def __init__(self):
        (self.train_X, self.train_Y), (self.test_X, self.test_Y) = mnist.load_data()

    def printFormat(self) :
        print('X_train: ' + str(self.train_X.shape))
        print('Y_train: ' + str(self.train_Y.shape))
        print('X_test:  '  + str(self.test_X.shape))
        print('Y_test:  '  + str(self.test_Y.shape))
    
    def printLabels(self):
        for i in range(sizeof(self.train_Y)): 
            print(self.train_Y[i])
        pyplot.show()
    
    def printMNIST(self, number):
        pyplot.imshow(self.train_X[number], cmap=pyplot.get_cmap('gray'))
        pyplot.show()

    def setStats(self) :
        table = np.array(self.train_X)
        pyplot.hist(table, self.train_Y)
        pyplot.title("Values Repartition")
        pyplot.show()
    
    def seabornStats(self):
        sns.set_theme(style="whitegrid")
        values= np.array(self.train_X)
        labels= np.array(self.train_Y)
        cols = pd.MultiIndex.from_product([values, labels])
        A = np.arange(1,9).reshape(2,2,2)
        df = pd.DataFrame(A.T.reshape(2,1), columns=cols)
        graph = sns.barplot(x="variable", data=df)
    #def changeImageFormat(self):

        

test = Bootstrap()

#test.printFormat()
test.setStats()
#test.seabornStats()
#test.printMNIST(4)
#test.printMNIST(2)