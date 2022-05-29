import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score, confusion_matrix
import mlflow
import mlflow.sklearn
from mlflow import log_metric, log_param, log_artifacts
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print('Starting the experiment')
    ##
   # mlflow.set_tracking_uri("http://127.0.0.1:5000")
   # mlflow.set_experiment(experiment_name='mlflow test demo-1')

    mlflow.autolog()  ##record automatically

    print('Loading the data')

    wine_df = pd.read_csv('data/winequality-red.csv', sep=';')

    # Combine 7&8 together; combine 3 and 4 with 5 so that we have only 3 levels and a more balanced Y variable
    wine_df['quality'] = wine_df['quality'].replace(8, 7)
    wine_df['quality'] = wine_df['quality'].replace(3, 5)
    wine_df['quality'] = wine_df['quality'].replace(4, 5)
    target_value_count = wine_df['quality'].value_counts()
    print(wine_df['quality'].value_counts())
    fig = plt.bar(list(target_value_count.index),list(target_value_count.values))
    plt.savefig("Target_Distribution.png")
    mlflow.log_artifact("Target_Distribution.png")

    log_param("Value counts", wine_df['quality'].value_counts())

    # splitting data into training and test set for independent attributes
    X_train, X_test, y_train, y_test = train_test_split(wine_df.drop('quality', axis=1), wine_df['quality'],
                                                        test_size=.4,
                                                        random_state=22)
    print(X_train.shape, X_test.shape)
    log_param("Train shape", X_train.shape)

    model_entropy = DecisionTreeClassifier(criterion="entropy",
                                           max_depth=1000, min_samples_leaf=15)

    model_entropy.fit(X_train, y_train)
    print("Model trained")

    train_accuracy = model_entropy.score(X_train, y_train)  # performance on train data
    test_accuracy = model_entropy.score(X_test, y_test)  # performance on test data

    log_metric("Accuracy for this run", test_accuracy)
    mlflow.sklearn.log_model(model_entropy, "Model")
    #mlflow.log_artifact('winequality-red.csv')
    # autolog_run = mlflow.last_active_run()
    print(mlflow.active_run().info.run_uuid)


