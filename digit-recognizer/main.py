import preprocess
from model import knn, mlp
import numpy as np
import pandas as pd

if __name__ == '__main__':

    print("Begin preprocessing data")
    X_train, y_train, X_validate, y_validate = preprocess.load_train_data()
    X_test = preprocess.load_test_data()
    print("Data preprocessing completed")

    model = "mlp"
    if model == 'knn':
        k = knn.train_knn(X_train, y_train, X_validate, y_validate)
        pred = knn.inference_knn(X_train, y_train, X_test, k)
    elif model == 'mlp':
        mlp_model = mlp.train_mlp(X_train, y_train)
        pred = mlp.inference_mlp(X_test, mlp_model)


    predictions_df = pd.DataFrame(data={'ImageId': np.arange(1, len(pred) + 1), 'Label': pred})
    
    if model == 'knn':
        predictions_df.to_csv('data/submission.csv', index=False)
    elif model == 'mlp':
        predictions_df.to_csv('data/submission_mlp.csv', index=False)
    print("Submission file created successfully")