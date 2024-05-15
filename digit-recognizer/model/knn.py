import numpy as np
import operator
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

def euc_dist(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    # K is the number of nearest neighbors to consider
    def __init__(self, K=3):
        self.K = K


    # fit method is used to train the model
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    # predict method is used to predict the output for the test data

    def predict(self, X_test):
        predictions = []
        for i in range(len(X_test)):
            dist = np.array([euc_dist(X_test[i], x_t) for x_t in self.X_train]) # calculate the distance of the test data with all the training data
            dist_sorted = dist.argsort()[:self.K] # sort the distance and get the indices of the K nearest neighbors
            neigh_count = {} # dictionary to store the count of each class in the K nearest neighbors
            for idx in dist_sorted: 
                if self.y_train[idx] in neigh_count:
                    neigh_count[self.y_train[idx]] += 1
                else:
                    neigh_count[self.y_train[idx]] = 1
            sorted_neigh_count = sorted(neigh_count.items(), key=operator.itemgetter(1), reverse=True)
            predictions.append(sorted_neigh_count[0][0])
        return np.array(predictions)


def train_knn(X_train, y_train, X_validate, y_validate):
    '''
    This function trains the KNN model on the training data and validates it on the validation data.
    '''
    print("Begin training")
    kVals = range(1, 5)
    accuracies = []
    for K in kVals:
        model = KNN(K)
        model.fit(X_train, y_train)
        pred = model.predict(X_validate)
        acc = accuracy_score(y_validate, pred)
        accuracies.append(acc)
        print("K: ", K, " Accuracy: ", acc)
    max_index = accuracies.index(max(accuracies))
    print("Best K: ", max_index + 1, " Accuracy: ", max(accuracies))
    # plt.plot(kVals, accuracies)
    # plt.xlabel('K Value')
    # plt.ylabel('Accuracy')
    print("Training completed")
    return max_index + 1

def inference_knn(X_train, y_train, X_test, k):
    '''
    This function predicts the output for the test data using the trained KNN model.
    '''
    print("Begin inference")
    model = KNN(k)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("Inference completed")
    return pred
