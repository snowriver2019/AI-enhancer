from keras.preprocessing.text import Tokenizer
import re
import itertools
from itertools import compress

def ToknSeq(kn,padtype):  
    bases=['A','T','G','C','X']
    kmers=[''.join(p) for p in itertools.product(bases, repeat=kn)]
    if padtype==1:
        vv=[not bool(re.search("[ATGC]X", i)) for i in kmers]
    else:
        vv=[not bool(re.search("X[ATGC]", i)) for i in kmers]
    kmers2=list(compress(kmers, vv))
    n=len(kmers2)
    Tokn=Tokenizer(num_words=n+1,lower=False)
    Tokn.fit_on_texts(kmers2)
    res=Tokn.index_word
    k=[k for (k, v) in res.items() if v == "X"*kn]
    return Tokn,int(k[0]),n

def encoderKmer(records,Tokn,padtype,kn):
    mat_seq=list()
    for rcd in records:
        sq=str(rcd.seq)
        if len(sq)<maxn:
            sq=(-len(sq)+maxn)*'X'+sq if padtype == 1 else sq+(-len(sq)+maxn)*'X'
        else:
            sq=sq[0:maxn]
        t=[sq[i:i+kn] for i in range(0,len(sq),kn)]
        mt=Tokn.texts_to_sequences(t)
        arr_pos1=np.array(mt)
        arr_pos2=arr_pos1.reshape(-1)
        mat_seq.append(arr_pos2.tolist())
    mat_all=np.array(mat_seq)
    return mat_all


