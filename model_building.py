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