import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq

tbs = pd.Series(list('AGCT'))
hot_table= pd.get_dummies(tbs)
hot_table['X']=[0,0,0,0]
hash_code= { x:y for x, y in zip(hot_table.columns.values,  hot_table.transpose().values) }
model=load_model("./Model/EpigeneticFeatureModel_epoch-08_acc-0.982-model2.h5")


def ecd_seq2(t):
    t3=t.apply(lambda x: [i for i in x])
    mat_pos=np.empty((0,4), int)
    for i in t3:
        m = [hash_code[c] for c in i]
        m1=np.array(m)
        mat_pos=np.vstack((mat_pos,m1))
    return(mat_pos)


def encoderFW(records,maxn,wnsz,stp):
    mat_seq=list()
    for rcd in records:
        sq0=str(rcd.seq)
        sqa=sq0.upper()
        sq1=sqa.replace("N","")
        if len(sq1)<maxn:
            sq=sq1+(-len(sq1)+maxn)*'X'
        else:
            sq=sq1[0:maxn]
        t=[sq[i:i+wnsz] for i in range(0, len(sq)-wnsz,stp)]
        mat_seq.append(t)
        return(mat_seq)

def encoderBW(records,maxn,wnsz,stp):
    mat_seq2=list()
    for rcd in records:
        sq0=str(rcd.seq)
        sqt=Seq(sq0)
        sq0 =str(sqt.reverse_complement())
        sqa=sq0.upper()
        sq1=sqa.replace("N","")
        if len(sq1)<maxn:
            sq=sq1+(-len(sq1)+maxn)*'X'
        else:
            sq=sq1[0:maxn]
        t=[sq[i:i+wnsz] for i in range(0, len(sq)-wnsz,stp)]
        mat_seq2.append(t)
        return(mat_seq2)

def encoder1hot(mat_seq):
    m_all=list()
    for k in mat_seq:
        mt=list()
        for xx in k:
            t=pd.DataFrame([xx])[0]
            m=ecd_seq2(t)  #.transpose()
            mt.append(m.tolist())
        m_all.append(mt)
    mat_all=np.array(m_all)
    return(mat_all)

def epiEncode(mat):
    mat_ecd=[]
    for mt in mat:
        yy=model.predict(mt)
        mat_ecd.append(yy)
    mat_ecd2=np.array(mat_ecd)
    return(mat_ecd2)

