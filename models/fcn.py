# FCN
import keras 
import numpy as np 
import pandas as pd 
import time 

from models.utils.utils import save_logs
from sklearn.preprocessing import LabelEncoder

class Classifier_FCN:

	def __init__(self, output_directory, input_shape, nb_classes, itr, verbose=False):
		self.output_directory = output_directory + "/"
		self.model = self.build_model(input_shape, nb_classes)
		if(verbose==True):
			self.model.summary()
		self.verbose = verbose
		self.iteration = itr
		self.model.save_weights(self.output_directory+'model_init.hdf5')

	def build_model(self, input_shape, nb_classes):
		input_layer = keras.layers.Input(input_shape)

		conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
		conv1 = keras.layers.normalization.BatchNormalization()(conv1)
		conv1 = keras.layers.Activation(activation='relu')(conv1)

		conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
		conv2 = keras.layers.normalization.BatchNormalization()(conv2)
		conv2 = keras.layers.Activation('relu')(conv2)

		conv3 = keras.layers.Conv1D(128, kernel_size=3,padding='same')(conv2)
		conv3 = keras.layers.normalization.BatchNormalization()(conv3)
		conv3 = keras.layers.Activation('relu')(conv3)

		gap_layer = keras.layers.pooling.GlobalAveragePooling1D()(conv3)

		output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

		model = keras.models.Model(inputs=input_layer, outputs=output_layer)
		model.summary()
		
		model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(), 
			metrics=['accuracy'])

		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, 
			min_lr=0.0001)

		file_path = self.output_directory+'best_model.hdf5'

		model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
			save_best_only=True)

		stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=0, mode='auto',
                                baseline=None, restore_best_weights=False)

		self.callbacks = [reduce_lr,model_checkpoint, stop_early]

		return model 

	def fit(self, x_train, y_train, x_val, y_val,y_true,batch_size=128,nb_epochs=1000): 
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
		# convert the predicted from binary to integer 
		y_pred = np.argmax(y_pred , axis=1)

		df_metrics = save_logs(self.output_directory, hist, y_pred, y_true, duration, pred_time, self.iteration)

		keras.backend.clear_session()

		return df_metrics

	
