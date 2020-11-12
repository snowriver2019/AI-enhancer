from Bio import SeqIO
from Bio.Seq import Seq
import numpy as np
import pandas as pd
import sys


rt="./Data/"
fl1=sys.argv[1]
fl2=sys.argv[2]
nm=sys.argv[3]

maxn=9000  
stp=200
wnsz=1000
records = list(SeqIO.parse(rt+fl1, "fasta"))
mat_seq = encoderFW(records,maxn,wnsz,stp)
mat_all_pos = encoder1hot(mat_seq)
np.save(rt+nm+'_pos_1hotFW.npy',mat_all_pos)
mat_seq = encoderBW(records,maxn,wnsz,stp)
mat_all_pos = encoder1hot(mat_seq)
np.save(rt+nm+'_pos_1hotBW.npy',mat_all_pos)

records = list(SeqIO.parse(rt+fl2, "fasta"))
mat_seq = encoderFW(records,maxn,wnsz,stp)
mat_all_neg=encoder1hot(mat_seq)
np.save(rt+nm+'_neg_1hotFW.npy',mat_all_neg)
mat_seq = encoderBW(records,maxn,wnsz,stp)
mat_all_neg = encoder1hot(mat_seq)
np.save(rt+nm+'_neg_1hotBW.npy',mat_all_neg)



