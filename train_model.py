import h5py
import numpy as np
import os


from keras.models import Sequential
from keras.layers import Dense,Masking
from keras.layers import Dropout
from keras.layers import GRU
from keras.layers import Embedding
from keras.models import load_model
from keras.callbacks import CSVLogger
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l1

def load_input(filename):
    with h5py.File( filename + '.hdf5', 'r') as hf:
        print("List of arrays in this file: \n", hf.keys())
        x = hf.get('input')
        y = hf.get('output')
        x_data = np.array(x)
        #n_patterns = x_data.shape[0]
        y_data = np.array(y)
        # print x_data[0][1000:1100]
        # print y_data[0][1000:1100]
        #y_data = y_data.reshape(y_data.shape+(1,))
        print(x_data.shape)
        print(y_data.shape)


    del x
    del y
    return x_data,y_data



def training1(storage,drop_out,batchsize,exp,data_x,data_y,cv_x,cv_y,gru_size1,gru_size2,epoch_size,reload = False,modelpath = None):

    class_weights = {0: 1, 1: 5}


    rmsprop = RMSprop(lr=0.002, rho=0.9, epsilon=1e-08, decay=0.0)

    timesteps = data_x.shape[1]
    feature_size =data_x.shape[-1]

    if not os.path.exists(storage):
        os.makedirs(storage)
    if reload == False:
        model = Sequential()
        model.add(Masking(mask_value=-1.0, input_shape=(timesteps, feature_size)))
        model.add(Dropout(drop_out))
        model.add(GRU(gru_size1,  return_sequences=True))
        model.add(GRU(gru_size2))
        model.add(Dense(1, activation='sigmoid',W_regularizer=l1(.1)))
        model.compile(loss='binary_crossentropy', optimizer=rmsprop,
                      metrics=['accuracy', 'fmeasure', 'precision', 'recall'])  # ,sample_weight_mode="temporal"

    else:
        model = load_model(modelpath)

    filepath = storage +"/weights-improvement-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=False, mode='min')
    csv_logger = CSVLogger('%s.csv' % exp)
    callbacks_list = [checkpoint, csv_logger]
    #hist = model.fit(train_x, train_y, nb_epoch=epoch_size, batch_size=batchsize, callbacks=callbacks_list, class_weight=class_weights, sample_weight=None)  # None)
    hist = model.fit(data_x, data_y, nb_epoch=epoch_size, batch_size=batchsize, callbacks=callbacks_list,validation_data=(cv_x,cv_y),class_weight=class_weights, sample_weight=None)  # None)
    model.save(storage + '/model_result.hdf5')
    np.save(storage + '/epoch_history.npy', hist.history)


#path = "/xdisk/dongfangxu9/sentiment_analysis/"
path = "features_all/"
features = "regex_matamap_embedding"

x_val, y_val= load_input(path+ "val_" +features)
x_train, y_train= load_input(path + "train_" +features)

exp = "exp_"+features
storage = path +features
epoch_size = 400
gru_size1 = 128
gru_size2 = 128
dropout = 0.25
batchsize = 100
training1(storage, dropout, batchsize, exp, x_train, y_train, x_val, y_val,gru_size1,gru_size2, epoch_size, reload=False,
          modelpath=None)
