import numpy as np
from queue import Queue

class TreeNode():
    def __init__(self, features, target, depth=0):
        self.left= None
        self.right= None
        self.threshold = None
        self.feature_index = None
        self.gain = 0.0
        self.has_child = False
        self.depth = depth
        self.features = features
        self.target = target

    def split_node(self, threshold, feature_index, gain):
        self.threshold = threshold
        self.feature_index = feature_index
        self.gain = gain

        left_indices = self.features[:, feature_index] > threshold
        right_indices = self.features[:, feature_index] <= threshold

        self.left = TreeNode(self.features[left_indices], self.target[left_indices], depth=self.depth+1)
        self.right = TreeNode(self.features[right_indices], self.target[right_indices], depth=self.depth+1)
        
        del self.features, self.target
        self.has_child = True
        self.features = None
        self.target = None


class DecisionTree():
    def __init__(self):
        self.root= None

    def fit(self, X, y):
        self.root = TreeNode(X, y)
        nodes = Queue()
        nodes.put(self.root)

        while nodes.qsize() > 0:
            current_node = nodes.get()
            threshold, feature_index, gain = self.find_best_gain(current_node.features, current_node.target)
            if gain > 0:
                current_node.split_node(threshold, feature_index, gain)
                if current_node.has_child:
                    nodes.put(current_node.left)
                    nodes.put(current_node.right)
        return self

    def predict(self, X):
        ret = []
        
        for sample in X:
            current_node= self.root
            while current_node.has_child:
                if sample[current_node.feature_index] > current_node.threshold:
                    current_node = current_node.left
                else:
                    current_node = current_node.right
            classes, counts = np.unique(current_node.target, return_counts=True)
            ret.append(classes[counts.argmax()])

        return np.array(ret)

    def gini_impurity(self, target, classes):
        ret = 1.0
        if len(target) == 0:
            return ret
        for cls in classes:
            ret -= (len(target[target == cls]) / len(target))**2
        return ret
        


    def compute_gain(self, feature, target, threshold):
        classes = set(target)
        criterion = self.gini_impurity
        

        target_left = target[feature > threshold]
        target_right = target[feature <= threshold]

        criterion_before = criterion(target, classes)
        criterion_left = criterion(target_left, classes)
        criterion_right = criterion(target_right, classes)
        criterion_after = ((criterion_left * len(target_left)) / len(target)) + ((criterion_right * len(target_right)) / len(target))
        
        gain = criterion_before - criterion_after
        return gain

    def find_best_gain(self, features, target):
        best_feature_index = -1
        best_gain = 0.0
        best_threshold = 0.0

        for feature_index in range(features.shape[1]):
            feature = features[:, feature_index]
            thresholds = list(set(feature))

            for threshold in thresholds:
                gain = self.compute_gain(feature, target, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_threshold, best_feature_index, best_gain


def train_dt(X_train, y_train):
    dt = DecisionTree()
    dt.fit(X_train, y_train)
    return dt

def inference_dt(X_test, dt):
    return dt.predict(X_test)


