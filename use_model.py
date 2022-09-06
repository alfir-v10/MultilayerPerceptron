from MultilayerPerceptron import MLP
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pylab as plt
import argparse


if __name__ == '__main__':
    pars = argparse.ArgumentParser()
    '''
        model = MLP(X_train, y_train, X_test, y_test, L=2, N_l=32)  
        model.train(batch_size=8, epochs=250, lr=0.01, early_stopping=5)
    '''
    pars.add_argument('--path', type=str,  help='path to data', default='data.csv')
    pars.add_argument('--mode', type=str, help='train or predict', default='train')
    pars.add_argument('--batch_size', type=int, help='batch_size', default=8)
    pars.add_argument('--epochs', type=int, help='epochs', default=20)
    pars.add_argument('--lr', type=float, help='learning rate', default=0.1)
    pars.add_argument('--early_stopping', type=int, help='early_stopping', default=2)
    pars.add_argument('--L', type=int, help='amount of hidden layers', default=2)
    pars.add_argument('--N_l', type=int, help='amount of neurons in layers', default=21)
    path = pars.parse_args().path
    mode = pars.parse_args().mode
    batch_size = pars.parse_args().batch_size
    epochs = pars.parse_args().epochs
    lr = pars.parse_args().lr
    early_stopping = pars.parse_args().early_stopping
    L = pars.parse_args().L
    N_l = pars.parse_args().N_l

    df = pd.read_csv(path, header=None)
    df.iloc[:, 1] = df.iloc[:, 1].replace({'M': 1, 'B': 0})
    df.drop_duplicates(inplace=True)
    X_data = df.iloc[:, 2:].values
    y_data = df.iloc[:, 1].values
    model_selection = ExtraTreesClassifier()
    model_selection.fit(X_data, y_data)
    feat_importances = pd.Series(model_selection.feature_importances_, index=df.iloc[:, 2:].columns)
    columns_to_drop = feat_importances.nsmallest(10).index.to_list()
    df = df.drop(columns_to_drop, axis=1)
    columns_to_drop = [27, 26, 25, 22, 9, 8, 7, 2, 5, 4, 12, 14]
    df = df.drop(columns_to_drop, axis=1)
    columns_to_train = df.columns
    df.iloc[:, 2:] = StandardScaler().fit_transform(df.iloc[:, 2:].values)
    X_data = df.iloc[:, 2:].values
    y_data = df.iloc[:, 1].values

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.1, random_state=42)

    if mode == 'train':
        model = MLP(X_train, y_train, X_test, y_test, L=L, N_l=N_l)
        model.train(batch_size=batch_size, epochs=epochs, lr=lr, early_stopping=early_stopping)
        model.save_model()

        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].plot(model.train_loss, label="Train loss")
        ax[0].plot(model.val_loss, label="Val loss")
        ax[0].legend()
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[0].grid()

        ax[1].plot(model.train_acc, label="Train acc")
        ax[1].plot(model.val_acc, label="Val acc")
        ax[1].legend()
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Accuracy")
        ax[1].grid()
        plt.savefig('loss.png')
        plt.show()

    if mode == 'predict':
        model = MLP()
        model.load_model()
        y_pred = model.predict(X_val)
        print("Accuracy Score: ", accuracy_score(y_true=y_val, y_pred=np.argmax(y_pred, axis=1)))

