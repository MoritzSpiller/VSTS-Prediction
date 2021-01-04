import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob

mpl.style.use('seaborn-paper')

from sklearn.preprocessing import LabelEncoder

import warnings
warnings.simplefilter('ignore', category=DeprecationWarning)

from keras.models import Model
from keras.layers import Permute
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import backend as K
import tensorflow as tf

def train_model(model:Model, X_train, y_train, X_test, y_test, epochs=50, batch_size=128, val_subset=None, from_ckpt=False,
                cutoff=None, normalize_timeseries=False, learning_rate=1e-3, monitor='loss', optimization_mode='auto', compile_model=True, fold=9):

    classes = np.unique(y_train)
    le = LabelEncoder()
    y_ind = le.fit_transform(y_train.ravel())
    recip_freq = len(y_train) / (len(le.classes_) * np.bincount(y_ind).astype(np.float64))
    class_weight = recip_freq[le.transform(classes)]

    print("Class weights : ", class_weight)

    y_train = to_categorical(y_train, len(np.unique(y_train)))
    y_test = to_categorical(y_test, len(np.unique(y_test)))

    factor = 1. / np.cbrt(2)

    weight_fn = os.path.join(Path(__file__).parents[0], "weights/weights.fold_%s_{epoch:02d}-{val_loss:.2f}.hdf5" % str(fold))

    model_checkpoint = ModelCheckpoint(weight_fn, verbose=1, mode=optimization_mode,
                                       monitor=monitor, save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=100, mode=optimization_mode,
                                  factor=factor, cooldown=0, min_lr=1e-4, verbose=2)
    stop_early = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=0, mode='auto',
                                baseline=None, restore_best_weights=False)
    callback_list = [model_checkpoint, reduce_lr, stop_early]

    optm = Adam(lr=learning_rate)

    if compile_model:
        model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

    if val_subset is not None:
        X_test = X_test[:val_subset]
        y_test = y_test[:val_subset]

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callback_list,
              class_weight=class_weight, verbose=2, validation_data=(X_test, y_test))


def evaluate_model(model:Model, X_test, y_test, batch_size=128, test_data_subset=None,
                   cutoff=None, normalize_timeseries=False, fold=9):

    y_test = to_categorical(y_test, len(np.unique(y_test)))

    optm = Adam(lr=1e-3)
    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy', f1_score, auroc])

    # find file with minimum validation loss for that fold
    parent_path = os.path.join(Path(__file__).parents[0], 'weights')
    # find all files for that fold
    file_list = glob(os.path.join(parent_path, 'weights.fold_{}*'.format(fold)))
    val_losses = []
    for f in file_list:
        val_losses.append(float(os.path.splitext(os.path.basename(f))[0].split('-')[-1]))

    min_loss_idx = val_losses.index(min(val_losses))

    weight_fn = file_list[min_loss_idx]
    model.load_weights(weight_fn)

    if test_data_subset is not None:
        X_test = X_test[:test_data_subset]
        y_test = y_test[:test_data_subset]

    print("\nEvaluating : ")
    loss, accuracy, f1, auc = model.evaluate(X_test, y_test, batch_size=batch_size)
    print()
    print("Final Accuracy : ", accuracy)

    return accuracy, loss, f1, auc

def set_trainable(layer, value):
   layer.trainable = value

   # case: container
   if hasattr(layer, 'layers'):
       for l in layer.layers:
           set_trainable(l, value)

   # case: wrapper (which is a case not covered by the PR)
   if hasattr(layer, 'layer'):
        set_trainable(layer.layer, value)

class MaskablePermute(Permute):

    def __init__(self, dims, **kwargs):
        super(MaskablePermute, self).__init__(dims, **kwargs)
        self.supports_masking = True


def f1_score(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)

    return 2 * ((precision * recall) / (precision + recall))

def auroc(y_true, y_pred):
    #We want strictly 1D arrays - cannot have (batch, 1), for instance
    true= K.flatten(y_true)
    pred = K.flatten(y_pred)

    #total number of elements in this batch
    totalCount = K.shape(true)[0]

    #sorting the prediction values in descending order
    values, indices = tf.nn.top_k(pred, k = totalCount)   
    #sorting the ground truth values based on the predictions above         
    sortedTrue = K.gather(true, indices)

    #getting the ground negative elements (already sorted above)
    negatives = 1 - sortedTrue

    #the true positive count per threshold
    TPCurve = K.cumsum(sortedTrue)

    #area under the curve
    auc = K.sum(TPCurve * negatives)

    #normalizing the result between 0 and 1
    totalCount = K.cast(totalCount, K.floatx())
    positiveCount = K.sum(true)
    negativeCount = totalCount - positiveCount
    totalArea = positiveCount * negativeCount

    return  auc / totalArea
