import keras
import warnings
import numpy as np
from keras.callbacks import Callback, ModelCheckpoint
import os
import glob

class keepbest(ModelCheckpoint):


    def __init__(self, filepath, monitor, verbose=0, mode='auto', period=1, \
        save_best_only=True,save_weights_only=True):
        super(keepbest, self).__init__('')
        self.filepath=filepath
        self.monitor = monitor
        self.verbose = verbose
        self.period = period
        self.best_epochs = 0
        self.epochs_since_last_save = 0
        self.save_best_only=save_best_only
        self.save_weights_only=save_weights_only


        if mode not in ['auto', 'min', 'max']:
            warnings.warn('GetBest mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf
                
    def on_train_begin(self, logs=None):
        self.best_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current

                        aa=filepath.split('-')

                        if self.save_weights_only:
                            fileList = glob.glob(aa[0]+"*"+aa[-1]+".hdf5", recursive=True)
                            for flnm in fileList:
                                try:
                                    os.remove(flnm)
                                except OSError:
                                    print("Error while deleting file")
             
                            self.model.save_weights(filepath+".hdf5", overwrite=True)
                        else:
                            fileList = glob.glob(aa[0]+"*"+aa[-1]+".h5", recursive=True)
                            for flnm in fileList:
                                try:
                                    os.remove(flnm)
                                except OSError:
                                    print("Error while deleting file")
                            self.model.save(filepath+".h5", overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True) 

 

    def on_train_end(self, logs=None):
        if self.verbose > 0:
            print('Using epoch %05d with %s: %0.5f' % (self.best_epochs, self.monitor,
                                                       self.best))
        self.model.set_weights(self.best_weights)







