from keras.layers import Input, Dense, Flatten, Dropout, Conv1D, concatenate, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.models import *

def CNNmodel(length,vocab_siz):
	inputs1 = Input(shape=(length,))
	embedding1 = Embedding(vocab_siz, 10)(inputs1)
	conv1 = Conv1D(filters=10, kernel_size=10, activation='sigmoid')(embedding1)
	drop1 = Dropout(0.5)(conv1)
	pool1 = MaxPooling1D(pool_size=10)(drop1)
	flat1 = Flatten()(pool1)

	conv2 = Conv1D(filters=20, kernel_size=100, activation='sigmoid')(embedding1)
	drop2 = Dropout(0.5)(conv2)
	pool2 = MaxPooling1D(pool_size=5)(drop2)
	flat2 = Flatten()(pool2)

	conv3 = Conv1D(filters=30, kernel_size=130, activation='sigmoid')(embedding1)
	drop3 = Dropout(0.5)(conv3)
	pool3 = MaxPooling1D(pool_size=5)(drop3)
	flat3 = Flatten()(pool3)

	merged = concatenate([flat1, flat2, flat3])
	outputs = Dense(1, activation='sigmoid')(merged)

	model = Model(inputs=inputs1, outputs=outputs)
	return model
