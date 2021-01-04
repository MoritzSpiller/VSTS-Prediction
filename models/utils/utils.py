from builtins import print
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import operator

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

from scipy.interpolate import interp1d
from scipy.io import loadmat

def calculate_metrics(y_true,y_pred,duration,pred_time,output_path,iteration,y_true_val=None,y_pred_val=None):
    res = pd.DataFrame(data = np.zeros((1,8),dtype=np.float), index=[0],
        columns=['precision','accuracy','recall','duration','f1_score','auc','loss','class_time'])
    precision = precision_score(y_true,y_pred,average='macro')
    recall = recall_score(y_true,y_pred,average='macro')
    res['precision'] = precision
    res['accuracy'] = accuracy_score(y_true,y_pred)
    res['f1_score'] = 2*(precision*recall) / (precision + recall)
    try:
        res['auc'] = roc_auc_score(y_true,y_pred,average='macro')
    except ValueError:
        res['auc'] = 0.0
    
    try:
        res['loss'] = log_loss(y_true,y_pred,eps=1e-15)
    except ValueError:
        res['loss'] = 0.0

    if not y_true_val is None:
        # this is useful when transfer learning is used with cross validation
        res['accuracy_val'] = accuracy_score(y_true_val,y_pred_val)

    res['recall'] = recall
    res['duration'] = duration
    res['class_time'] = pred_time

    if len(np.unique(y_true)) == 2:
        fpr,tpr,threshold = roc_curve(y_true,y_pred)
        roc_auc = auc(fpr,tpr)
        plot_roc_auc(fpr,tpr,roc_auc,output_path,iteration)

    return res

def plot_roc_auc(fpr,tpr,roc_auc,path,itr):
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    
    plt.savefig(os.path.join(path,'roc_curve_{}.png'.format(itr)))

def transform_labels(y_train,y_test,y_val=None):
    """
    Transform label to min equal zero and continuous
    For example if we have [1,3,4] --->  [0,1,2]
    """
    if not y_val is None :
        # index for when resplitting the concatenation
        idx_y_val = len(y_train)
        idx_y_test = idx_y_val + len(y_val)
        # init the encoder
        encoder = LabelEncoder()
        # concat train and test to fit
        y_train_val_test = np.concatenate((y_train,y_val,y_test),axis =0)
        # fit the encoder
        encoder.fit(y_train_val_test)
        # transform to min zero and continuous labels
        new_y_train_val_test = encoder.transform(y_train_val_test)
        # resplit the train and test
        new_y_train = new_y_train_val_test[0:idx_y_val]
        new_y_val = new_y_train_val_test[idx_y_val:idx_y_test]
        new_y_test = new_y_train_val_test[idx_y_test:]
        return new_y_train, new_y_val,new_y_test
    else:
        # no validation split
        # init the encoder
        encoder = LabelEncoder()
        # concat train and test to fit
        y_train_test = np.concatenate((y_train,y_test),axis =0)
        # fit the encoder
        encoder.fit(y_train_test)
        # transform to min zero and continuous labels
        new_y_train_test = encoder.transform(y_train_test)
        # resplit the train and test
        new_y_train = new_y_train_test[0:len(y_train)]
        new_y_test = new_y_train_test[len(y_train):]
        return new_y_train, new_y_test

def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_'+metric])
    plt.title('model '+metric)
    plt.ylabel(metric,fontsize='large')
    plt.xlabel('epoch',fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name,bbox_inches='tight')
    plt.close()

def save_logs(output_directory, hist, y_pred, y_true,duration,pred_time,iteration,lr=True,y_true_val=None,y_pred_val=None):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory+'history_{}.csv'.format(iteration), index=False)

    df_metrics = calculate_metrics(y_true,y_pred,duration,pred_time,output_directory,iteration,y_true_val,y_pred_val)
    df_metrics.to_csv(output_directory+'df_metrics_{}.csv'.format(iteration), index=False)

    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data = np.zeros((1,6),dtype=np.float) , index = [0],
        columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
        'best_model_val_acc', 'best_model_learning_rate','best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['acc']
    df_best_model['best_model_val_acc'] = row_best_model['val_acc']
    if lr == True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory+'df_best_model_{}.csv'.format(iteration), index=False)

    # for FCN there is no hyperparameters fine tuning - everything is static in code

    # plot losses
    plot_epochs_metric(hist, output_directory+'epochs_loss_{}.png'.format(iteration))

    return df_metrics