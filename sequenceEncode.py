import sys
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from processingFuncs import encoderFW, encoder1hot, epiEncode




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
mat_ps = epiEncode(mat_all_pos)
np.save(rt+nm+"_pos_epiFt.npy",mat_ps)

records = list(SeqIO.parse(rt+fl2, "fasta"))
mat_seq = encoderFW(records,maxn,wnsz,stp)
mat_all_neg=encoder1hot(mat_seq)
mat_ng = epiEncode(mat_all_neg)
np.save(rt+nm+"_neg_epiFt.npy",mat_ng)


