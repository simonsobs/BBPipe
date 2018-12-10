import numpy as np
import sacc
import matplotlib.pyplot as plt

tracer_names=np.array(["BK15_95","BK15_150","BK15_220",
                       "W023","P030","W033","P044",
                       "P070","P100","P143","P217","P353"])
exp_names=np.array(["BicepKeck","BicepKeck","BicepKeck",
                    "WMAP","Planck","WMAP","Planck",
                    "Planck","Planck","Planck","Planck","Planck"])
corr_ordering=np.array(
    [['BK15_95_E','BK15_95_E'],['BK15_95_B','BK15_95_B'],['BK15_150_E','BK15_150_E'],['BK15_150_B','BK15_150_B'],['BK15_220_E','BK15_220_E'],['BK15_220_B','BK15_220_B'],['W023_E','W023_E'],['W023_B','W023_B'],['P030_E','P030_E'],['P030_B','P030_B'],['W033_E','W033_E'],['W033_B','W033_B'],['P044_E','P044_E'],['P044_B','P044_B'],['P070_E','P070_E'],['P070_B','P070_B'],['P100_E','P100_E'],['P100_B','P100_B'],['P143_E','P143_E'],['P143_B','P143_B'],['P217_E','P217_E'],['P217_B','P217_B'],['P353_E','P353_E'],['P353_B','P353_B'],['BK15_95_E','BK15_95_B'],['BK15_95_B','BK15_150_E'],['BK15_150_E','BK15_150_B'],['BK15_150_B','BK15_220_E'],['BK15_220_E','BK15_220_B'],['BK15_220_B','W023_E'],['W023_E','W023_B'],['W023_B','P030_E'],['P030_E','P030_B'],['P030_B','W033_E'],['W033_E','W033_B'],['W033_B','P044_E'],['P044_E','P044_B'],['P044_B','P070_E'],['P070_E','P070_B'],['P070_B','P100_E'],['P100_E','P100_B'],['P100_B','P143_E'],['P143_E','P143_B'],['P143_B','P217_E'],['P217_E','P217_B'],['P217_B','P353_E'],['P353_E','P353_B'],['BK15_95_E','BK15_150_E'],['BK15_95_B','BK15_150_B'],['BK15_150_E','BK15_220_E'],['BK15_150_B','BK15_220_B'],['BK15_220_E','W023_E'],['BK15_220_B','W023_B'],['W023_E','P030_E'],['W023_B','P030_B'],['P030_E','W033_E'],['P030_B','W033_B'],['W033_E','P044_E'],['W033_B','P044_B'],['P044_E','P070_E'],['P044_B','P070_B'],['P070_E','P100_E'],['P070_B','P100_B'],['P100_E','P143_E'],['P100_B','P143_B'],['P143_E','P217_E'],['P143_B','P217_B'],['P217_E','P353_E'],['P217_B','P353_B'],['BK15_95_E','BK15_150_B'],['BK15_95_B','BK15_220_E'],['BK15_150_E','BK15_220_B'],['BK15_150_B','W023_E'],['BK15_220_E','W023_B'],['BK15_220_B','P030_E'],['W023_E','P030_B'],['W023_B','W033_E'],['P030_E','W033_B'],['P030_B','P044_E'],['W033_E','P044_B'],['W033_B','P070_E'],['P044_E','P070_B'],['P044_B','P100_E'],['P070_E','P100_B'],['P070_B','P143_E'],['P100_E','P143_B'],['P100_B','P217_E'],['P143_E','P217_B'],['P143_B','P353_E'],['P217_E','P353_B'],['BK15_95_E','BK15_220_E'],['BK15_95_B','BK15_220_B'],['BK15_150_E','W023_E'],['BK15_150_B','W023_B'],['BK15_220_E','P030_E'],['BK15_220_B','P030_B'],['W023_E','W033_E'],['W023_B','W033_B'],['P030_E','P044_E'],['P030_B','P044_B'],['W033_E','P070_E'],['W033_B','P070_B'],['P044_E','P100_E'],['P044_B','P100_B'],['P070_E','P143_E'],['P070_B','P143_B'],['P100_E','P217_E'],['P100_B','P217_B'],['P143_E','P353_E'],['P143_B','P353_B'],['BK15_95_E','BK15_220_B'],['BK15_95_B','W023_E'],['BK15_150_E','W023_B'],['BK15_150_B','P030_E'],['BK15_220_E','P030_B'],['BK15_220_B','W033_E'],['W023_E','W033_B'],['W023_B','P044_E'],['P030_E','P044_B'],['P030_B','P070_E'],['W033_E','P070_B'],['W033_B','P100_E'],['P044_E','P100_B'],['P044_B','P143_E'],['P070_E','P143_B'],['P070_B','P217_E'],['P100_E','P217_B'],['P100_B','P353_E'],['P143_E','P353_B'],['BK15_95_E','W023_E'],['BK15_95_B','W023_B'],['BK15_150_E','P030_E'],['BK15_150_B','P030_B'],['BK15_220_E','W033_E'],['BK15_220_B','W033_B'],['W023_E','P044_E'],['W023_B','P044_B'],['P030_E','P070_E'],['P030_B','P070_B'],['W033_E','P100_E'],['W033_B','P100_B'],['P044_E','P143_E'],['P044_B','P143_B'],['P070_E','P217_E'],['P070_B','P217_B'],['P100_E','P353_E'],['P100_B','P353_B'],['BK15_95_E','W023_B'],['BK15_95_B','P030_E'],['BK15_150_E','P030_B'],['BK15_150_B','W033_E'],['BK15_220_E','W033_B'],['BK15_220_B','P044_E'],['W023_E','P044_B'],['W023_B','P070_E'],['P030_E','P070_B'],['P030_B','P100_E'],['W033_E','P100_B'],['W033_B','P143_E'],['P044_E','P143_B'],['P044_B','P217_E'],['P070_E','P217_B'],['P070_B','P353_E'],['P100_E','P353_B'],['BK15_95_E','P030_E'],['BK15_95_B','P030_B'],['BK15_150_E','W033_E'],['BK15_150_B','W033_B'],['BK15_220_E','P044_E'],['BK15_220_B','P044_B'],['W023_E','P070_E'],['W023_B','P070_B'],['P030_E','P100_E'],['P030_B','P100_B'],['W033_E','P143_E'],['W033_B','P143_B'],['P044_E','P217_E'],['P044_B','P217_B'],['P070_E','P353_E'],['P070_B','P353_B'],['BK15_95_E','P030_B'],['BK15_95_B','W033_E'],['BK15_150_E','W033_B'],['BK15_150_B','P044_E'],['BK15_220_E','P044_B'],['BK15_220_B','P070_E'],['W023_E','P070_B'],['W023_B','P100_E'],['P030_E','P100_B'],['P030_B','P143_E'],['W033_E','P143_B'],['W033_B','P217_E'],['P044_E','P217_B'],['P044_B','P353_E'],['P070_E','P353_B'],['BK15_95_E','W033_E'],['BK15_95_B','W033_B'],['BK15_150_E','P044_E'],['BK15_150_B','P044_B'],['BK15_220_E','P070_E'],['BK15_220_B','P070_B'],['W023_E','P100_E'],['W023_B','P100_B'],['P030_E','P143_E'],['P030_B','P143_B'],['W033_E','P217_E'],['W033_B','P217_B'],['P044_E','P353_E'],['P044_B','P353_B'],['BK15_95_E','W033_B'],['BK15_95_B','P044_E'],['BK15_150_E','P044_B'],['BK15_150_B','P070_E'],['BK15_220_E','P070_B'],['BK15_220_B','P100_E'],['W023_E','P100_B'],['W023_B','P143_E'],['P030_E','P143_B'],['P030_B','P217_E'],['W033_E','P217_B'],['W033_B','P353_E'],['P044_E','P353_B'],['BK15_95_E','P044_E'],['BK15_95_B','P044_B'],['BK15_150_E','P070_E'],['BK15_150_B','P070_B'],['BK15_220_E','P100_E'],['BK15_220_B','P100_B'],['W023_E','P143_E'],['W023_B','P143_B'],['P030_E','P217_E'],['P030_B','P217_B'],['W033_E','P353_E'],['W033_B','P353_B'],['BK15_95_E','P044_B'],['BK15_95_B','P070_E'],['BK15_150_E','P070_B'],['BK15_150_B','P100_E'],['BK15_220_E','P100_B'],['BK15_220_B','P143_E'],['W023_E','P143_B'],['W023_B','P217_E'],['P030_E','P217_B'],['P030_B','P353_E'],['W033_E','P353_B'],['BK15_95_E','P070_E'],['BK15_95_B','P070_B'],['BK15_150_E','P100_E'],['BK15_150_B','P100_B'],['BK15_220_E','P143_E'],['BK15_220_B','P143_B'],['W023_E','P217_E'],['W023_B','P217_B'],['P030_E','P353_E'],['P030_B','P353_B'],['BK15_95_E','P070_B'],['BK15_95_B','P100_E'],['BK15_150_E','P100_B'],['BK15_150_B','P143_E'],['BK15_220_E','P143_B'],['BK15_220_B','P217_E'],['W023_E','P217_B'],['W023_B','P353_E'],['P030_E','P353_B'],['BK15_95_E','P100_E'],['BK15_95_B','P100_B'],['BK15_150_E','P143_E'],['BK15_150_B','P143_B'],['BK15_220_E','P217_E'],['BK15_220_B','P217_B'],['W023_E','P353_E'],['W023_B','P353_B'],['BK15_95_E','P100_B'],['BK15_95_B','P143_E'],['BK15_150_E','P143_B'],['BK15_150_B','P217_E'],['BK15_220_E','P217_B'],['BK15_220_B','P353_E'],['W023_E','P353_B'],['BK15_95_E','P143_E'],['BK15_95_B','P143_B'],['BK15_150_E','P217_E'],['BK15_150_B','P217_B'],['BK15_220_E','P353_E'],['BK15_220_B','P353_B'],['BK15_95_E','P143_B'],['BK15_95_B','P217_E'],['BK15_150_E','P217_B'],['BK15_150_B','P353_E'],['BK15_220_E','P353_B'],['BK15_95_E','P217_E'],['BK15_95_B','P217_B'],['BK15_150_E','P353_E'],['BK15_150_B','P353_B'],['BK15_95_E','P217_B'],['BK15_95_B','P353_E'],['BK15_150_E','P353_B'],['BK15_95_E','P353_E'],['BK15_95_B','P353_B'],['BK15_95_E','P353_B']])

def get_tracer_from_name(name,exp_sample=None) :
    d=np.loadtxt("BK15_cosmomc/data/BK15/bandpass_"+name+".txt",unpack=True)
    return sacc.Tracer(name,"spin2",d[0],d[1],exp_sample)

#Tracers
tracers=[get_tracer_from_name(t,e) for t,e in zip(tracer_names,exp_names)]

#Mean vector
dv=np.loadtxt("BK15_cosmomc/data/BK15/BK15_cl_hat.dat",unpack=True)[1:]
nells=dv.shape[-1]
meanvec=sacc.MeanVec(dv.flatten())

#Precision matrix
precis=sacc.Precision(np.loadtxt("BK15_cosmomc/data/BK15/BK15_covmat_dust.dat",unpack=True))

#Binning
ls=np.loadtxt("BK15_cosmomc/data/BK15/windows/BK15_bpwf_bin1.txt",unpack=True)[0]
windows=np.array([np.loadtxt("BK15_cosmomc/data/BK15/windows/BK15_bpwf_bin%d.txt"%(i+1),unpack=True)[1:]
                  for i in range(nells)])
typ_arr=[]
ls_arr=[]
t1_arr=[]
t2_arr=[]
q1_arr=[]
q2_arr=[]
w_arr=[]
for ic,c in enumerate(corr_ordering) :
    s1,s2=c
    tn1=s1[:-2]
    q1=s1[-1]
    t1=np.where(tracer_names==tn1)[0][0]
    tn2=s2[:-2]
    q2=s2[-1]
    t2=np.where(tracer_names==tn2)[0][0]
    typ=q1+q2
    for b in range(nells) :
        w=windows[b,ic]
        lmean=np.sum(ls*w)/np.sum(w)
        win=sacc.Window(ls,w)
        ls_arr.append(lmean)
        w_arr.append(win)
    q1_arr+=nells*[q1]
    q2_arr+=nells*[q2]
    t1_arr+=nells*[t1]
    t2_arr+=nells*[t2]
    typ_arr+=nells*[typ]
bins=sacc.Binning(typ_arr,ls_arr,t1_arr,q1_arr,t2_arr,q2_arr,windows=w_arr)

#SACC file
s=sacc.SACC(tracers,bins,mean=meanvec,precision=precis,meta={'data_name':'BK15_bmode_analysis'})

#Save SACC file
s.saveToHDF("BK15.sacc")
s.printInfo()
