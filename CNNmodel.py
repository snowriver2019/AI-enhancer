length=lens
vocab_siz=nvocab
inputs1 = Input(shape=(length,))
embedding1 = Embedding(vocab_siz, 2)(inputs1)
conv1 = Conv1D(filters=10, kernel_size=10, activation='sigmoid')(embedding1)
drop1 = Dropout(0.5)(conv1)
pool1 = MaxPooling1D(pool_size=10)(drop1)
flat1 = Flatten()(pool1)
# channel 2
inputs2 = Input(shape=(length,))
embedding2 = Embedding(vocab_siz, 2)(inputs2)
conv2 = Conv1D(filters=20, kernel_size=100, activation='sigmoid')(embedding2)
drop2 = Dropout(0.5)(conv2)
pool2 = MaxPooling1D(pool_size=5)(drop2)
flat2 = Flatten()(pool2)
# channel 3
inputs3 = Input(shape=(length,))
embedding3 = Embedding(vocab_siz, 2)(inputs3)
conv3 = Conv1D(filters=15, kernel_size=130, activation='sigmoid')(embedding3)
drop3 = Dropout(0.5)(conv3)
pool3 = MaxPooling1D(pool_size=5)(drop3)
flat3 = Flatten()(pool3)
# merged = concatenate([flat1, flat2])
merged = concatenate([flat1, flat2, flat3])
# interpretation
# dense1 = Dense(10, activation='relu')(merged)
outputs = Dense(1, activation='sigmoid')(merged)
model = Model(inputs=[inputs1,inputs2,inputs3], outputs=outputs)
# compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# summarize
model.summary()