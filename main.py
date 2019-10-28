import numpy as np
import pandas as pd
import processingFuncs
import model_building
from Bio import SeqIO
from keras.preprocessing import sequence
from callback_keep_latest import keepbest


kn = 6  ### the length of kmer
maxn=9000 ### maximum length of a sequence
padtype=1  ### 1 = padding 0 before the sequence, 0 = padding 0 after the sequence

###### Get Tokenizer based on the kmer sequence-word #################
Tokn,kstop,vocab_siz=ToknSeq(kn,padtype)

####### Read the data and encode the data #############
records = list(SeqIO.parse("./humanEHC_VISTA-ext_posSet.fa", "fasta"))
mat_all_pos=encoderKmer(records,Tokn,padtype)
records = list(SeqIO.parse("./humanEHC_VISTA-ext_negSet.fa", "fasta"))
mat_all_neg=encoderKmer(records,Tokn,padtype)

mat_all_pos[mat_all_pos==kstop]=0
mat_all_neg[mat_all_neg==kstop]=0

np.save('EHR_VISTA_pos_6mer_noOvlap_padEd.npy', mat_all_pos)
np.save('EHR_VISTA_neg_6mer_noOvlap_padEd.npy', mat_all_neg)

###### build model ###################
seqlen=mat_all_pos.shape[1]
model=CNNmodel(seqlen,vocab_siz)

filepath = "./Model/EHR_epoch-{epoch:02d}_acc-{val_acc:.2f}-CNNmodel"
checkpoint = keepbest(filepath=filepath, monitor='val_acc', verbose=1, \
     save_best_only=True, mode='max',save_weights_only=True)
callbacks_list = [checkpoint]
model.fit(X_train, Y_train, epochs=100, batch_size=10 , callbacks=callbacks_list,
          validation_data=(X_test, Y_test),verbose=2)

