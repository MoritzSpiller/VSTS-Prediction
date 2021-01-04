# ResNet
# when tuning start with learning rate->mini_batch_size -> 
# momentum-> #hidden_units -> # learning_rate_decay -> #layers 
import keras 
import numpy as np 
import pandas as pd 
import time

import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt 

from models.utils.utils import save_logs
from sklearn.preprocessing import LabelEncoder

class Classifier_RESNET: 

	def __init__(self, output_directory, input_shape, nb_classes, itr, verbose=False):
		self.output_directory = output_directory + "/"
		self.model = self.build_model(input_shape, nb_classes)
		if(verbose==True):
			self.model.summary()
		self.verbose = verbose
		self.iteration = itr
		self.model.save_weights(self.output_directory+'model_init.hdf5')

	def build_model(self, input_shape, nb_classes):
		n_feature_maps = 64

		input_layer = keras.layers.Input(input_shape)
		
		# BLOCK 1 

		conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
		conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
		conv_x = keras.layers.Activation('relu')(conv_x)

		conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
		conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
		conv_y = keras.layers.Activation('relu')(conv_y)

		conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
		conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

		# expand channels for the sum 
		shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
		shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

		output_block_1 = keras.layers.add([shortcut_y, conv_z])
		output_block_1 = keras.layers.Activation('relu')(output_block_1)

		# BLOCK 2 

		conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_1)
		conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
		conv_x = keras.layers.Activation('relu')(conv_x)

		conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_x)
		conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
		conv_y = keras.layers.Activation('relu')(conv_y)

		conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_y)
		conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

		# expand channels for the sum 
		shortcut_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1, padding='same')(output_block_1)
		shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

		output_block_2 = keras.layers.add([shortcut_y, conv_z])
		output_block_2 = keras.layers.Activation('relu')(output_block_2)

		# BLOCK 3 

		conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_2)
		conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
		conv_x = keras.layers.Activation('relu')(conv_x)

		conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_x)
		conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
		conv_y = keras.layers.Activation('relu')(conv_y)

		conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_y)
		conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

		# no need to expand channels because they are equal 
		shortcut_y = keras.layers.normalization.BatchNormalization()(output_block_2)

		output_block_3 = keras.layers.add([shortcut_y, conv_z])
		output_block_3 = keras.layers.Activation('relu')(output_block_3)

		# FINAL 
		
		gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

		output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

		model = keras.models.Model(inputs=input_layer, outputs=output_layer)
		model.summary()
		
		model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), 
			metrics=['accuracy'])

		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

		file_path = self.output_directory+'best_model.hdf5' 

		model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
			save_best_only=True)

		stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=0, mode='auto',
                                baseline=None, restore_best_weights=False)

		self.callbacks = [reduce_lr,model_checkpoint,stop_early]

		return model
	
	def fit(self, x_train, y_train, x_val, y_val, y_true, batch_size=128, nb_epochs=1000): 
		# x_val and y_val are only used to monitor the test loss and NOT for training  

		# mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

		classes = np.unique(y_train)
		le = LabelEncoder()
		y_ind = le.fit_transform(y_train.ravel())
		recip_freq = len(y_train) / (len(le.classes_) * np.bincount(y_ind).astype(np.float64))
		class_weights = recip_freq[le.transform(classes)]

		start_time = time.time() 

		hist = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epochs,
			verbose=self.verbose, validation_data=(x_val,y_val), callbacks=self.callbacks, class_weight=class_weights)

		duration = time.time() - start_time

		model = keras.models.load_model(self.output_directory+'best_model.hdf5')

		start_pred = time.time()
		y_pred = model.predict(x_val)
		pred_time = time.time() - start_pred
		print(y_pred.shape)
		# convert the predicted from binary to integer 
		y_pred = np.argmax(y_pred , axis=1)

		df_metrics = save_logs(self.output_directory, hist, y_pred, y_true, duration, pred_time, self.iteration)

		keras.backend.clear_session()

		return df_metrics
