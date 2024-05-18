import numpy as np
from queue import Queue
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

class TreeNode():
    '''
    A node in the decision tree
    '''
    def __init__(self, X_train, y_train, depth=0):
        '''
        Initialize the node with the given features and target 
        '''
        self.left= None
        self.right= None
        self.threshold = None
        self.feature_index = None
        self.gain = 0.0
        self.has_child = False
        self.depth = depth
        self.features = X_train
        self.target = y_train

    def split_node(self, threshold, feature_index, gain):
        '''
        Split the node into two child nodes
        '''
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
    '''
    A simple decision tree classifier
    '''
    def __init__(self):
        self.root= None

    def fit(self, X_train, y_train):
        '''
        Fit the decision tree with the given training data
        '''
        self.root = TreeNode(X_train, y_train)
        nodes = Queue()
        nodes.put(self.root)

        while nodes.qsize() > 0:
            current_node = nodes.get()
            threshold, feature_index, gain = self.find_best_gain(current_node.features, current_node.target)
            if gain > 0: # If the gain is greater than 0, split the node
                current_node.split_node(threshold, feature_index, gain)
                if current_node.has_child:
                    nodes.put(current_node.left)
                    nodes.put(current_node.right)
        return self

    def predict(self, X_test):
        '''
        Predict the target for the given test data
        '''
        ret = []
        
        for sample in X_test:
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
        '''
        Compute the gini impurity of the target, the lower the value, the better the split
        '''
        ret = 1.0
        if len(target) == 0:
            return ret
        for cls in classes:
            ret -= (len(target[target == cls]) / len(target))**2
        return ret
        


    def compute_gain(self, feature, target, threshold):
        '''
        Compute the gain of the split using the gini impurity
        '''
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
        '''
        Find the best gain for the given features and target
        '''
        best_feature_index = -1
        best_gain = 0.0
        best_threshold = 0.0

        for feature_index in range(features.shape[1]): # iterate over all features
            feature = features[:, feature_index] # All samples for the current feature
            thresholds = list(set(feature)) # All unique values for the current feature

            for threshold in thresholds:
                gain = self.compute_gain(feature, target, threshold) # Compute the gain for the current feature and threshold
                if gain > best_gain:
                    best_gain = gain
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_threshold, best_feature_index, best_gain


def train_dt(X_train, y_train, X_validate, y_validate):
    '''
    Train the decision tree with the given training data
    '''
    dt = DecisionTree()
    dt.fit(X_train, y_train)

    # Validate the model
    pred = dt.predict(X_validate)
    acc = accuracy_score(y_validate, pred)
    print("Accuracy: ", acc)
    bacc = balanced_accuracy_score(y_validate, pred)
    print("Balanced Accuracy: ", bacc)
    prec = precision_score(y_validate, pred, average=None)
    print("Precision: ", prec)
    rec = recall_score(y_validate, pred, average=None)
    print("Recall: ", rec)
    plot(prec, rec)
    return dt

def inference_dt(X_test, dt):
    '''
    Predict the target for the given test data
    '''
    return dt.predict(X_test)


def plot(precisions, recalls):
    # Define classes (0 to 9)
    classes = list(range(10))

    # Create subplots for precision and recall
    fig, axs = plt.subplots(figsize=(10, 8))

    # Plot precision by class
    axs.bar(classes, precisions, color='blue')
    axs.set_title('Precision by Class')
    axs.set_xlabel('Class')
    axs.set_ylabel('Precision')

    # Save precision plot
    plt.tight_layout()
    plt.savefig('log/decisiontree/precision_by_class.png', dpi=300)
    plt.close()



    # Create new figure for recall
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot recall by class
    ax.bar(classes, recalls, color='green')
    ax.set_title('Recall by Class')
    ax.set_xlabel('Class')
    ax.set_ylabel('Recall')

    # Save recall plot
    plt.tight_layout()
    plt.savefig('log/decisiontree/recall_by_class.png', dpi=300)
    plt.close()



