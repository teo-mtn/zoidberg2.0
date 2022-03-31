from keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

class DataSets:

    def __init__(self):
        self.k_range = range(1,26)
        self.scores = {}
        self.scores_list = []
        (self.train_X, self.train_y), (self.test_X, self.test_y) = mnist.load_data()

    def print_sets(self):
        print('X_train: ' + str(self.train_X.shape))
        print('Y_train: ' + str(self.train_y.shape))
        print('X_test:  ' + str(self.test_X.shape))
        print('Y_test:  ' + str(self.test_y.shape))
        
    def show_plot(self, nb_plot):
        plt.imshow(self.train_X[nb_plot], cmap=plt.get_cmap('gray'))
        plt.show()

    def showStatPixels(self, nb_plot):
        nb_label = self.train_y[nb_plot]
        plt.hist(self.train_X[nb_plot], label=nb_label)
        plt.xlabel("Color")
        plt.ylabel("Number occurence")
        plt.title(nb_label)
        plt.show()
        
    def showStatsLabel(self):
        plt.hist(self.train_y)
        plt.show()

    def resize_image(self, number):
        table = np.array(self.train_X)
        table_1d = table[number].flatten()
        print(table_1d.shape)
    
    def printPandasTable(self) :
        train=self.train_y
        test=self.test_y
        df = pd.DataFrame({"train_x":pd.DataFrame(train).value_counts(), "test_x":pd.DataFrame(test).value_counts()})
        print(df)

    def seabornGraph(self):
        df = pd.DataFrame(self.train_y)
        ax = sb.histplot(data=df, palette="plasma")
        plt.show()
        
    def reshape_array(self):
        self.train_X = self.train_X.reshape(60000, 28*28)
        self.train_y = self.train_y.flatten()

        
        self.test_X = self.test_X.reshape(10000, 28*28)
        self.train_y = self.train_y.flatten()
        
    def find_best_neighbor_number(self):
        for k in self.k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(self.train_X, self.train_y)
            y_pred = knn.predict(self.test_X.reshape(10000, 28 * 28))
            self.scores[k] = metrics.accuracy_score(self.test_y, y_pred)
            self.scores_list.append(metrics.accuracy_score(self.test_y, y_pred))

    def print_score(self):
        print(self.scores)
        print(self.scores_list)

    def train_IA(self):
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(self.train_X, self.train_y)
        y_pred = knn.predict(self.test_X.reshape(10000, 28 * 28))
        for i in range(0,100):
            print(self.test_y[i], ":", y_pred[i])
        return y_pred
        
    def plot_accuracy(self, prediction):
        plt.plot(self.k_range, self.scores_list)
        plt.xlabel("Value in image")
        plt.ylabel("Number of Occurences")
        plt.show()
        
    def hist_values_of_test(self, prediction):
        plt.plot(prediction)
        plt.show()
        

ds = DataSets()
#ds.print_sets()
# ds.show_plot(2)
# ds.showStatPixels(200)
#ds.showStatsLabel()
#ds.resize_image(3)
#ds.printPandasTable()
#ds.seabornGraph()
#ds.train_IA()

ds.reshape_array()
#ds.find_best_neighbor_number()
#ds.print_score()
#ds.plot_accuracy()
pred = ds.train_IA()
#ds.hist_values_of_test(pred)

