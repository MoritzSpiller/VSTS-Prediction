import keras 
import numpy as np 
import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder

from keras.models import Model
from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from models.utils.keras_utils import train_model, evaluate_model, set_trainable
from models.utils.layer_utils import AttentionLSTM
from models.utils.utils import save_logs

class Classifier_MLSTM:

    def __init__(self, output_directory, input_shape, nb_classes, itr, verbose=False):
        self.output_directory = output_directory + "/"
        self.model = self.build_malstm(input_shape, nb_classes)
        if(verbose==True):
            self.model.summary()
        self.verbose = verbose
        self.iteration = itr
        self.model.save_weights(self.output_directory+'model_init.hdf5')

    def build_mlstm(self, input_shape, nb_classes):
        ip = Input(shape=input_shape)

        x = Masking()(ip)
        x = LSTM(4)(x)
        x = Dropout(0.8)(x)

        y = Permute((2, 1))(ip)
        y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = self.squeeze_excite_block(y)

        y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = self.squeeze_excite_block(y)

        y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = GlobalAveragePooling1D()(y)

        x = concatenate([x, y])

        out = Dense(nb_classes, activation='softmax')(x)

        model = Model(ip, out)
        model.summary()

        # add load model code here to fine-tune

        return model


    def build_malstm(self, input_shape, nb_classes):
        ip = Input(shape=input_shape)

        x = Masking()(ip)
        x = AttentionLSTM(4)(x)
        x = Dropout(0.8)(x)

        y = Permute((2, 1))(ip)
        y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = self.squeeze_excite_block(y)

        y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = self.squeeze_excite_block(y)

        y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = GlobalAveragePooling1D()(y)

        x = concatenate([x, y])

        out = Dense(nb_classes, activation='softmax')(x)

        model = Model(ip, out)
        model.summary()

        model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
        
        file_path = self.output_directory+'best_model.hdf5'
        
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)

        factor = 1. / np.cbrt(2)
        reduce_lr = ReduceLROnPlateau(monitor='loss', patience=50, mode='auto', factor=factor, cooldown=0, min_lr=1e-4, verbose=2)
        stop_early = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=0, mode='auto', baseline=None, restore_best_weights=False)

        self.callbacks = [reduce_lr,stop_early,model_checkpoint]
        # add load model code here to fine-tune

        return model

    def squeeze_excite_block(self, input):
        ''' Create a squeeze-excite block
        Args:
            input: input tensor
            filters: number of output filters
            k: width factor

        Returns: a keras tensor
        '''
        filters = input._keras_shape[-1] # channel_axis = -1 for TF

        se = GlobalAveragePooling1D()(input)
        se = Reshape((1, filters))(se)
        se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
        se = multiply([input, se])

        return se

    def fit(self, x_train, y_train, x_val, y_val,y_true,batch_size=128,nb_epochs=1000,val_subset=None,
            from_ckpt=False, cutoff=None, normalize_timeseries=False, learning_rate=1e-3, monitor='loss', optimization_mode='auto', compile_model=True): 
        # x_val and y_val are only used to monitor the test loss and NOT for training

        classes = np.unique(y_train)
        le = LabelEncoder()
        y_ind = le.fit_transform(y_train.ravel())
        recip_freq = len(y_train) / (len(le.classes_) * np.bincount(y_ind).astype(np.float64))
        class_weights = recip_freq[le.transform(classes)]

        start_time = time.time() 

        hist = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epochs,
            verbose=self.verbose, validation_data=(x_val,y_val), callbacks=self.callbacks, class_weight=class_weights)

        duration = time.time() - start_time

        model = keras.models.load_model(self.output_directory+'best_model.hdf5', custom_objects={"AttentionLSTM" : AttentionLSTM})

        start_pred = time.time()
        y_pred = model.predict(x_val)
        pred_time = time.time() - start_pred
        # convert the predicted from binary to integer 
        y_pred = np.argmax(y_pred, axis=1)

        df_metrics = save_logs(self.output_directory, hist, y_pred, y_true, duration, pred_time, self.iteration)

        keras.backend.clear_session()

        return df_metrics
