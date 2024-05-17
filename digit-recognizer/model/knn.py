import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist

def euc_dist(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    def __init__(self, K=3):
        '''
        This is the constructor of the KNN class.
        K is the number of nearest neighbors to consider
        '''
        self.K = K


    def fit(self, X_train, y_train):
        '''
        Read the training data and store it in the object
        '''
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        '''
        This function predicts the output for the test data.
        '''
        predictions = []
        dists = cdist(X_test, self.X_train, metric='euclidean') # Calculate the distance between the test data and all the training data
        
        for dist in dists: # for each test data
            dist_sorted_indices = dist.argsort()[:self.K]  # Sort the distance and get the indices of the K nearest neighbors
            neigh_count = {}  # Store the count of each class in the K nearest neighbors

            for idx in dist_sorted_indices:
                label = self.y_train[idx]
                if label in neigh_count:
                    neigh_count[label] += 1
                else:
                    neigh_count[label] = 1

            # Sort the dictionary based on the count of each class
            sorted_neigh_count = sorted(neigh_count.items(), key=lambda item: item[1], reverse=True)
            predictions.append(sorted_neigh_count[0][0])

        return np.array(predictions)


def train_knn(X_train, y_train, X_validate, y_validate):
    '''
    This function trains the KNN model on the training data and validates it on the validation data.
    '''
    print("Begin training")
    kVals = range(1, 11)
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
    plt.plot(kVals, accuracies)
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.title('K-Accuracy Plot')
    plt.xticks(kVals)
    plt.savefig('log/knn/k_accuracy_plot.png')
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

