import glob
import sys
import numpy as np
import pandas as pd
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Dense, Activation, LSTM, Bidirectional, Lambda,Input,concatenate,subtract, \
 multiply, maximum,Permute,RepeatVector,
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
from callback_keep_latest import keepbest



def attentionModel(ntime,nft):

    inputs1 = Input(shape=(ntime,nft))
        
    bilstm = Bidirectional(LSTM(120, return_sequences=True), 
                            input_shape=(ntime, nft))(inputs1)
    
    slc1 = Lambda(lambda x: x[:,0,:])(bilstm)
    slc2 = Lambda(lambda x: x[:,-1,:])(bilstm)

    attention1 = Dense(1, activation='tanh')(bilstm)
    attention1 = Flatten()(attention1)
    attention1 = Activation('softmax')(attention1)
    attention1 = RepeatVector(240)(attention1)
    attention1 = Permute([2, 1])(attention1)
    repst1 = multiply([attention1,bilstm])

    sent_repst1 = Lambda(lambda x: K.sum(x, axis=1))(repst1)
    comb=concatenate([slc1,slc2,sent_repst1])

    dns=Dense(900)(comb)
    outputs = Dense(1, activation='sigmoid')(dns)
    model = Model(inputs=[inputs1], outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return(model)


rt="./EncodeData"
filelist=sys.argv[1]
nm=sys.argv[2]

list_table = pd.read_csv(rt+filelist,delimiter="\n" , header=None)
cellns=list_table[0].values
k=1
for cel in cellns:
    fls=glob.glob(rt+'*'+cel+'*.npy')
    if fls:
        f1=glob.glob(rt+'*'+cel+'*pos_epiFt.npy')[0]
        f2=glob.glob(rt+'*'+cel+'*neg_epiFt.npy')[0] 
        mat_pos=np.load(f1)
        mat_neg=np.load(f2)
        mat_t=np.vstack((mat_pos,mat_neg))
        labl_yt=np.append(np.repeat(1, mat_pos.shape[0]),np.repeat(0,mat_neg.shape[0]))
        if(k==1):
            mat_all=np.empty((0,mat_pos.shape[1],mat_pos.shape[2]))
            labl_yy=np.empty((0,1))
            k=2
        mat_all=np.vstack((mat_all,mat_t))
        labl_yy=np.append(labl_yy,labl_yt)

X_train, X_test, ts_train,ts_test,Y_train, Y_test ,indx_train,indx_test = train_test_split(mat_all, labl_yy,test_size=0.3)
ntime=X_train.shape[1]
nft=X_train.shape[2]

model=attentionModel(ntime,nft)

filepath = rt+"AttentionCombModel_epoch-{epoch:02d}_acc-{val_acc:.2f}-" + nm
checkpoint = keepbest(filepath=filepath, monitor='val_acc', verbose=1, \
     save_best_only=True, mode='max',save_weights_only=False)
earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
callbacks_list = [checkpointer,earlystopper]

model.fit(X_train, Y_train, epochs=56, batch_size=50, callbacks=callbacks_list,
          validation_data=(X_test, Y_test),verbose=2)


