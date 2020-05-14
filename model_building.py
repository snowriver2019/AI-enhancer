from keras.preprocessing import sequence
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, LSTM, Bidirectional,Input, multiply, maximum
from keras.layers.embeddings import Embedding


def model_build(ntime,nft):

    inputs1 = Input(shape=(ntime,nft))
    inputs2 = Input(shape=(ntime,nft))
    inputs3 = Input(shape=(1,))
        
    bilstm = Bidirectional(LSTM(120, return_sequences=False), 
                            input_shape=(ntime, nft))(inputs1)
        
    bilstm2 = Bidirectional(LSTM(120, return_sequences=False), 
                            input_shape=(ntime, nft))(inputs2)

    bil=maximum([bilstm, bilstm2])

    emb = Embedding(6, 240, input_length=1)(inputs3)
    fln = Flatten()(emb)
    comb = multiply([bil,fln])
    outputs = Dense(1, activation='sigmoid')(comb)
    model = Model(inputs=[inputs1,inputs2,inputs3], outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model


