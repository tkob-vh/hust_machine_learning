import preprocess
import knn.knn as knn
import numpy as np
import pandas as pd

if __name__ == '__main__':
    X_train, y_train, X_validate, y_validate = preprocess.load_train_data()
    X_test = preprocess.load_test_data()

    k = knn.train_knn(X_train, y_train, X_validate, y_validate)
    pred = knn.inference_knn(X_train, y_train, X_test, k)


    predictions_df = pd.DataFrame(data={'ImageId': np.arange(1, len(pred) + 1), 'Label': pred})
    predictions_df.to_csv('data/submission.csv', index=False)
    print("Submission file created successfully")