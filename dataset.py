import os
from glob import glob
import numpy as np
from pathlib import Path
from models.utils.utils import transform_labels
import pandas as pd
import sklearn

class dataset:
    # load dataset
    # k-fold cross-validation
    # scale
    # documentation

    def __init__(self, k_folds, path=None):
        self.folds = k_folds
        self.data, self.labels = self.load_dataset(path)
        self.metrics = pd.DataFrame(data =None,
                columns=['fold_no', 'precision','accuracy','recall','duration','f1_score','auc','loss','class_time'])

    def load_dataset(self, path):
        if path is None:
            dataset_path = os.path.join(Path(__file__).parents[0], 'dataset')
        else:
            dataset_path = path

        x_data = self.scale(np.load(os.path.join(dataset_path, 'data.npy')))
        y_labels = np.load(os.path.join(dataset_path, 'labels.npy'))

        # COMMENTED FOR HUGE DATASETS - shuffle data
        #x_new, y_new = self.shuffle(x_data, y_labels)

        return x_data, y_labels

    def shuffle(self, x, y):
        shuffled_x = np.zeros(x.shape, dtype=float)
        shuffled_y = np.zeros(y.shape, dtype=int)
        idx_ary = np.transpose(np.array([np.linspace(0,x.shape[0]-1, x.shape[0]), np.linspace(0,y.shape[0]-1, y.shape[0])], dtype=int))
        np.random.shuffle(idx_ary)
        for new_idx, pair in enumerate(idx_ary):
            shuffled_x[new_idx,:,:] = x[pair[0],:,:]
            shuffled_y[new_idx] = y[pair[1]]

        if self.check_distribution(shuffled_y) == False:
            self.shuffle(shuffled_x, shuffled_y)

        return shuffled_x, shuffled_y

    def check_distribution(self, ary):
        # divide set into folds
        step = int(ary.shape[0]/self.folds)

        start_idx = 0
        for i in range(1,self.folds+1):
            if i == self.folds:
                end_idx = ary.shape[0]-1
            else:
                end_idx = step * i

            num_min_class = np.amin(np.bincount(ary[start_idx:end_idx])) 
            if num_min_class == 0:
                return False

            start_idx = end_idx+1

        return True

    def scale(self,data):
        scalers = {}
        for i in range(data.shape[1]):
            scalers[i] = sklearn.preprocessing.MinMaxScaler()
            data[:, i, :] = scalers[i].fit_transform(data[:, i, :]) 

        return data

    def split_folds(self, fold_no):
        sequences_per_fold = int(self.data.shape[0] / self.folds)

        idx_start_test_set = (fold_no-1) * sequences_per_fold 
        idx_end_test_set = idx_start_test_set + sequences_per_fold
        if fold_no == 1:
            x_train = self.data[idx_end_test_set:,:,:]
            x_test = self.data[idx_start_test_set:idx_end_test_set,:,:]
            y_train = self.labels[idx_end_test_set:]
            y_test = self.labels[idx_start_test_set:idx_end_test_set]
        elif fold_no == self.folds:
            x_train = self.data[0:idx_start_test_set,:,:]
            x_test = self.data[idx_start_test_set:,:,:]
            y_train = self.labels[0:idx_start_test_set]
            y_test = self.labels[idx_start_test_set:]
        else:
            x_train = self.data[np.r_[0:idx_start_test_set,idx_end_test_set:self.data.shape[0]]]
            x_test = self.data[idx_start_test_set:idx_end_test_set,:,:]
            y_train = self.labels[np.r_[0:idx_start_test_set,idx_end_test_set:self.labels.shape[0]]]
            y_test = self.labels[idx_start_test_set:idx_end_test_set]

        y_train, y_test = transform_labels(y_train, y_test)

        return x_train, x_test, y_train, y_test

    def save_performance(self, fold_no, performance_data, output_path):
        if fold_no < self.folds:
            performance_data.insert(0, 'fold_no', fold_no)
            copy_perf = performance_data.copy()
            self.metrics = self.metrics.append(performance_data, ignore_index=True)

        else:
            # add last performance data to df
            performance_data.insert(0, 'fold_no', fold_no)
            self.metrics = self.metrics.append(performance_data, ignore_index=True)

            # calculate averages and stddev
            df_avg_stddev = pd.DataFrame(data=np.zeros((2,9),dtype=np.float), index=[0,1],
            columns=['fold_no', 'precision','accuracy','recall','duration','f1_score','auc','loss','class_time'])

            df_avg_stddev['fold_no'].iloc[0] = 'mean'
            df_avg_stddev['precision'].iloc[0] = self.metrics['precision'].mean()
            df_avg_stddev['accuracy'].iloc[0] = self.metrics['accuracy'].mean()
            df_avg_stddev['recall'].iloc[0] = self.metrics['recall'].mean()
            df_avg_stddev['duration'].iloc[0] = self.metrics['duration'].mean()
            df_avg_stddev['f1_score'].iloc[0] = self.metrics['f1_score'].mean()
            df_avg_stddev['auc'].iloc[0] = self.metrics['auc'].mean()
            df_avg_stddev['loss'].iloc[0] = self.metrics['loss'].mean()
            df_avg_stddev['class_time'].iloc[0] = self.metrics['class_time'].mean()

            df_avg_stddev['fold_no'].iloc[1] = 'stddev'
            df_avg_stddev['precision'].iloc[1] = self.metrics['precision'].std()
            df_avg_stddev['accuracy'].iloc[1] = self.metrics['accuracy'].std()
            df_avg_stddev['recall'].iloc[1] = self.metrics['recall'].std()
            df_avg_stddev['duration'].iloc[1] = self.metrics['duration'].std()
            df_avg_stddev['f1_score'].iloc[1] = self.metrics['f1_score'].std()
            df_avg_stddev['auc'].iloc[1] = self.metrics['auc'].std()
            df_avg_stddev['loss'].iloc[1] = self.metrics['loss'].std()
            df_avg_stddev['class_time'].iloc[1] = self.metrics['class_time'].std()

            self.metrics = self.metrics.append(df_avg_stddev, ignore_index=True)

            # save performance data
            self.metrics.to_csv(path_or_buf=os.path.join(output_path, 'performance_summary.csv'),sep=',',index=False)
