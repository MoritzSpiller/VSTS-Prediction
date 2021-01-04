import numpy as np
import sys
import sklearn
from sklearn.preprocessing import LabelBinarizer
from models.utils.utils import transform_labels
from pathlib import Path
import os
import statistics
import csv
import datetime
import argparse
from glob import glob
import pandas as pd
from models.fcn import Classifier_FCN as fcn
from models.resnet import Classifier_RESNET as resnet
from models.mlstmfcn import Classifier_MLSTM as malstm
from dataset import dataset

def fit_classifier(classifier_str, k, epochs, batch_size):
    output_directory = os.path.join(Path(__file__).parents[0], "logs", classifier_str)

    # load dataset
    data = dataset(k)

    for i in range(1,data.folds+1):
        # get current train and test folds
        x_train, x_test, y_train, y_test = data.split_folds(i)

        # save orignal y because later we will use binary
        y_true = y_test.astype(np.int64)

        nb_classes = len(np.unique(y_train))

        if nb_classes == 2:
            #* for binary classification
            # transform the labels from integers to one hot vectors
            enc = sklearn.preprocessing.OneHotEncoder()
            enc.fit(np.concatenate((y_train,y_test),axis=0).reshape(-1,1))
            y_train = enc.transform(y_train.reshape(-1,1)).toarray()
            y_test = enc.transform(y_test.reshape(-1,1)).toarray()
        else:
            #* for multiclass classification
            lb = LabelBinarizer()
            lb.fit(np.concatenate((y_train,y_test),axis=0).reshape(-1,1))
            LabelBinarizer(neg_label=0,pos_label=1,sparse_output=False)
            y_train = lb.transform(y_train.reshape(-1,1))
            y_test = lb.transform(y_test.reshape(-1,1))

        input_shape = x_train.shape[1:]
        if classifier_str == 'mlstm':
            classifier = malstm(output_directory, input_shape, nb_classes, i, verbose=False)
        elif classifier_str == 'resnet':
            classifier = resnet(output_directory, input_shape, nb_classes, i, verbose=False)
        elif classifier_str == 'fcn':
            classifier = fcn(output_directory, input_shape, nb_classes, i, verbose=False)
        else:
            print("Check spelling!")
            print("Exiting!")
            exit

        df_metrics = classifier.fit(x_train, y_train, x_test, y_test, y_true, batch_size=batch_size, nb_epochs=epochs)

        data.save_performance(i, df_metrics, output_directory)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-cl", "--classifier", dest="classifier", default="mlstm", help="which classifier to use. mlstm (default), resnet or fcn")
    parser.add_argument("-k", "--k-folds", dest="k", default=10, help="number of folds for k-fold cross-validation. default = 10")
    parser.add_argument("-ep", "--nb_epochs", dest="nb_epochs", default=1000, help="number of epochs. default = 1000")
    parser.add_argument("-bs","--batch_size",dest="batch_size",default=32,help="specify batch size. default = 32")
    
    args = parser.parse_args()

    fit_classifier(args.classifier, int(args.k), int(args.nb_epochs), int(args.batch_size))