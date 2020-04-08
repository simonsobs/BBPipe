import matplotlib.pyplot as plt
import numpy as np

ells = np.loadtxt('ell_b.txt')

bbdata_sd1300 = np.loadtxt('bbdata_cls_1300.txt')
model_sd1300 = np.loadtxt('model_cls_1300.txt')

bbdata_sd1301 = np.loadtxt('bbdata_cls_1301.txt')
model_sd1301 = np.loadtxt('model_cls_1301.txt')

bbdata_sd1302 = np.loadtxt('bbdata_cls_1302.txt')
model_sd1302 = np.loadtxt('model_cls_1302.txt')

bbdata_sd1303 = np.loadtxt('bbdata_cls_1303.txt')
model_sd1303 = np.loadtxt('model_cls_1303.txt')

bbdata_sd1304 = np.loadtxt('bbdata_cls_1304.txt')
model_sd1304 = np.loadtxt('model_cls_1304.txt')

for i in range(6):
    for j in range(6):
        plt.figure()
        plt.plot(ells, bbdata_sd1300[i::6,j][:27],'k-', label='data sim 1')
        plt.plot(ells, model_sd1300[i::6,j][:27], 'k--', label='model sim 1')
        plt.plot(ells, bbdata_sd1301[i::6,j][:27],'r-', label='data sim 2')
        plt.plot(ells, model_sd1301[i::6,j][:27],'r--', label='model sim 2')
        plt.plot(ells, bbdata_sd1302[i::6,j][:27], 'b-', label='data sim 3')
        plt.plot(ells, model_sd1302[i::6,j][:27], 'b--', label='model sim 3')
        plt.plot(ells, bbdata_sd1303[i::6,j][:27],'c-', label='data sim 4')
        plt.plot(ells, model_sd1303[i::6,j][:27],'c--', label='model sim 4')
        plt.plot(ells, bbdata_sd1304[i::6,j][:27], 'm-', label='data sim 5')
        plt.plot(ells, model_sd1304[i::6,j][:27], 'm--', label='model sim 5')
        plt.title(f'$D_\\ell$_band{i}_x_band{j}',fontsize=14)
        plt.yscale('log')
        plt.xlabel('$\\ell$',fontsize=15)
        plt.ylabel('$D_\\ell$',fontsize=15)
        plt.legend()
        plt.savefig(f'./cls_model_vs_data_{i}_{j}_momF.png',bbox_inches='tight')

