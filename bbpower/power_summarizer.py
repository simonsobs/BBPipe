from bbpipe import PipelineStage
from .types import TextFile, SACCFile,DirFile
import sacc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

class BBPowerSummarizer(PipelineStage):
    name="BBPowerSummarizer"
    inputs=[('splits_list',TextFile),('bandpasses_list',TextFile),('cells_fiducial',SACCFile),
            ('cells_all_splits',SACCFile),('cells_all_sims',TextFile)]
    outputs=[('cell_plots',DirFile),('cells_coadded_total',SACCFile),('cells_coadded',SACCFile),
             ('cells_noise',SACCFile),('cells_null',SACCFile)]
    config_options={'nulls_covar_type':'diagonal',
                    'nulls_covar_diag_order': 0,
                    'data_covar_type':'block_diagonal',
                    'data_covar_diag_order': 3,
                    'do_plots': True}
    
    def save_figure(self,plot_title,extension='png'):
        fname=self.get_output('cell_plots')+'/'+plot_title+'.'+extension
        print(fname)
        plt.savefig(fname,bbox_inches='tight')

    def get_covariance_from_samples(self,v,covar_type='dense',
                                    off_diagonal_cut=0):
        """
        Computes a covariance matrix from a set of samples in the form [nsamples, ndata]
        """
        if covar_type=='diagonal':
            cov=np.std(v,axis=0)**2
            return sacc.Precision(matrix=cov,is_covariance=True,mode="diagonal")
        else:
            nsim, nd = v.shape
            vmean = np.mean(v,axis=0) 
            cov = np.einsum('ij,ik',v,v)
            cov = cov/nsim - vmean[None,:]*vmean[:,None]
            if covar_type=='block_diagonal':
                nblocks = nd // self.n_bpws
                cuts = np.ones([self.n_bpws, self.n_bpws])
                if nblocks * self.n_bpws != nd:
                    raise ValueError("Vector can't be divided into blocks")
                for i in range(off_diagonal_cut+1,self.n_bpws):
                    cuts -= np.diag(np.ones(self.n_bpws-i),k=i)
                    cuts -= np.diag(np.ones(self.n_bpws-i),k=-i)
                cov = cov.reshape([nblocks, self.n_bpws, nblocks, self.n_bpws])
                cov = (cov * cuts[None, :, None, :]).reshape([nd, nd])
            return sacc.Precision(matrix=cov,is_covariance=True,mode="dense")

    def save_to_sacc(self,fname,t,b,v,cov=None,return_sacc=False):
        s=sacc.SACC(t,b,mean=v,precision=cov)
        s.saveToHDF(fname)
        if return_sacc:
            return s

    def init_params(self):
        """
        Read some input files to determine the size of the power spectra
        """
        # Open plots directory
        if not os.path.isdir(self.get_output('cell_plots')):
            os.mkdir(self.get_output('cell_plots'))

        # Calculate number of splits and number of frequency channels
        self.nsplits=len(open(self.get_input('splits_list'),'r').readlines())
        self.nbands=len(open(self.get_input('bandpasses_list'),'r').readlines())

        # Compute all possible null combinations
        # Currently we compute these as (m_i-m_j) x (m_k-m_l)
        # where m_x is the set of maps for split x,
        # (i,j,k,l) are all different numbers.
        # Note that the actual number of possible nulls is actually infinite.
        # The most general form would be (sum_i a_i * m_i) x (sum_i b_i * m_i)
        # where both weight vectors a_i, b_i are orthogonal and have zero sum.
        # This way we're restricting ourselves to cases of the form:
        #    a = (1,-1,0,0), b=(0,0,1,-1)
    
        # First, figure out all possible pairings
        first_pairs=[]
        self.pairings=[]
        for i in range(self.nsplits):
            for j in list(filter(lambda x : x not in [i], range(self.nsplits))): # Loop over js that aren't i
                if j<i:
                    continue
                first_pairs.append((i,j))
                for k in list(filter(lambda x : x not in [i,j], range(self.nsplits))): # ks that aren't j or i
                    for l in list(filter(lambda x : x not in [i,j,k], range(self.nsplits))): # l != i,j,k
                        if l<k:
                            continue
                        if (k,l) in first_pairs:
                            continue
                        self.pairings.append((i,j,k,l))
        self.n_nulls=len(self.pairings)

        # First, initialize n_bpws to zero
        self.n_bpws = 0
        self.sorting = None
        # Read splits power spectra
        self.s_splits=sacc.SACC.loadFromHDF(self.get_input('cells_all_splits'))
        # Read sorting and number of bandpowers
        self.check_sacc_consistency(self.s_splits)
        # Read file names for the power spectra of all simulations
        with open(self.get_input('cells_all_sims')) as f:
            content=f.readlines()
        self.fname_sims=[x.strip() for x in content]
        self.nsims=len(self.fname_sims)
        # Polarization indices and names
        self.index_pol={'E':0,'B':1}
        self.pol_names=['E','B']

    def check_sacc_consistency(self,s):
        """
        Checks the consistency of the SACC file and returns number of
        expected bandpowers.
        """
        bands=[]
        splits=[]
        for t in s.tracers:
            band,split=t.name[2:-1].split('_',2) # Tracer names are bandX_splitY
            bands.append(band)
            splits.append(split)
        bands=np.unique(bands)
        splits=np.unique(splits)
        if ((len(bands)!=self.nbands) or (len(splits)!=self.nsplits) or
            (len(s.tracers)!=self.nbands*self.nsplits)):
            raise ValueError("There's something wrong with these SACC tracers")
        if self.sorting is None:
            self.sorting=s.sortTracers()
            # Number of bandpowers
            _, _, _, self.ells, _ = self.sorting[0]
            self.n_bpws=len(self.ells)

        # Total number of power spectra expected
        nx_expected=((self.nbands*self.nsplits*2)*(self.nbands*self.nsplits*2+1))//2
        nv_expected=self.n_bpws*nx_expected
        if (len(self.sorting) != nx_expected) or (len(s.mean.vector)!=nv_expected):
            raise ValueError("There's something wrong with the SACC binnign or mean")

    def get_tracers(self,s):
        """
        Gets two array of tracers: one for coadd SACC files, one for null SACC files.
        """
        tracers_bands={}
        for t in s.tracers:
            band,split=t.name[2:-1].split('_',2)
            if split=='split1':
                T=sacc.Tracer(band,t.type,t.z,t.Nz,exp_sample=t.exp_sample)
                T.addColumns({'dnu':t.extra_cols['dnu']})
                tracers_bands[band]=T
        if False:#self.config['do_plots']:
            plt.figure()
            for b in range(self.nbands):
                t=tracers_bands['band%d'%(b+1)]
                plt.plot(t.z,t.Nz,label='band %d'%(b+1))
            plt.xlabel('$\\nu [{\\rm GHz}]$',fontsize=15)
            plt.ylabel('Bandpass transmission',fontsize=15)
            self.save_figure('bandpasses')
            plt.close()

        self.t_coadd=[]
        for i in range(self.nbands):
            self.t_coadd.append(tracers_bands['band%d'%(i+1)])

        self.t_nulls=[]
        self.ind_nulls={}
        ind_null=0
        for b in range(self.nbands):
            t=tracers_bands['band%d'%(b+1)]
            for i in range(self.nsplits): # Loop over unique pairs
                for j in range(i,self.nsplits):
                    name='band%d_null%dm%d'%(b+1,i+1,j+1)
                    self.ind_nulls[name]=ind_null
                    T=sacc.Tracer(name,t.type,t.z,t.Nz,exp_sample=t.exp_sample)
                    T.addColumns({'dnu':t.extra_cols['dnu']})
                    self.t_nulls.append(T)
                    ind_null+=1

    def get_binnings(self,with_windows=True):
        # Get windows if needed
        win_coadd=None
        win_nulls=None
        if with_windows:
            win_coadd=[]
            win_nulls=[]
            ls_win = self.s_splits.binning.windows[0].ls.copy()
            nls=len(ls_win)
            windows=np.zeros([self.nbands,2,self.nbands,2,self.n_bpws,nls])
            for t1,t2,typ,ells,ndx in self.sorting:
                b1,s1=self.tracer_number_to_band_split(t1)
                b2,s2=self.tracer_number_to_band_split(t2)
                typ=typ.decode()
                p1=self.index_pol[typ[0]]
                p2=self.index_pol[typ[1]]
                if (s1==0) and (s2==0):
                    for b,i in enumerate(ndx):
                        w=self.s_splits.binning.windows[i].w
                        windows[b1,p1,b2,p2,b,:]=w
                        if not ((b1==b2) and (p1==p2)):
                            windows[b2,p2,b1,p1,b,:]=w

        # Binnings for coadds
        typ, ell, t1, q1, t2, q2 = [], [], [], [], [], []
        for i1 in range(2*self.nbands):
            b1=i1//2
            p1=i1%2
            for i2 in range(i1,2*self.nbands):
                b2=i2//2
                p2=i2%2
                ty=self.pol_names[p1]+self.pol_names[p2]
                for il,ll in enumerate(self.ells):
                    ell.append(ll)
                    typ.append(ty)
                    t1.append(b1)
                    t2.append(b2)
                    q1.append('C')
                    q2.append('C')
                    if with_windows:
                        win_coadd.append(sacc.Window(ls_win,windows[b1,p1,b2,p2,il]))
        self.bins_coadd=sacc.Binning(typ,ell,t1,q1,t2,q2,windows=win_coadd)

        # Binnings for nulls
        typ, ell, t1, q1, t2, q2 = [], [], [], [], [], []
        for i_null,(i,j,k,l) in enumerate(self.pairings):
            for b1 in range(self.nbands):
                tr1=self.ind_nulls['band%d_null%dm%d'%(b1+1,i+1,j+1)]
                for p1 in range(2):
                    for b2 in range(self.nbands):
                        tr2=self.ind_nulls['band%d_null%dm%d'%(b2+1,k+1,l+1)]
                        for p2 in range(2):
                            ty=self.pol_names[p1]+self.pol_names[p2]
                            for il,ll in enumerate(self.ells):
                                ell.append(ll)
                                typ.append(ty)
                                t1.append(tr1)
                                t2.append(tr2)
                                q1.append('C')
                                q2.append('C')
                                if with_windows:
                                    win_nulls.append(sacc.Window(ls_win,windows[b1,p1,b2,p2,il]))
        self.bins_nulls=sacc.Binning(typ,ell,t1,q1,t2,q2,windows=win_nulls)
        
    def tracer_number_to_band_split(self,itracer):
        """
        Translates between tracer number in splits file and
        band and split numbers
        """
        split=itracer%self.nsplits
        band=itracer//self.nsplits
        return band,split

    def parse_splits_sacc_file(self,s,plot_stuff=False):
        """
        Transform a SACC file containing splits into 4 SACC vectors:
        1 that contains the coadded power spectra.
        1 that contains coadded power spectra for cross-split only.
        1 that contains an estimate of the noise power spectrum.
        1 that contains all null tests
        """

        # Check we have the right number of bands, splits, cross-correlations and power spectra
        self.check_sacc_consistency(s)

        # Now read power spectra into an array of form [nsplits,nsplits,nbands,nbands,2,2,n_ell]
        # This duplicates the number of elements, but simplifies bookkeeping significantly.
        spectra=np.zeros([self.nsplits,self.nsplits,
                          self.nbands,2,self.nbands,2,
                          self.n_bpws])
        for t1,t2,typ,ells,ndx in self.sorting:
            # Band, split and polarization channel indices
            b1, s1 = self.tracer_number_to_band_split(t1)
            b2, s2 = self.tracer_number_to_band_split(t2)
            typ=typ.decode()
            p1=self.index_pol[typ[0]]
            p2=self.index_pol[typ[1]]
            is_x = not ((b1==b2) and (s1==s2) and (p1==p2))
            spectra[s1,s2,b1,p1,b2,p2,:]=s.mean.vector[ndx]
            if is_x:
                spectra[s2,s1,b2,p2,b1,p1,:]=s.mean.vector[ndx]

        # Coadding (assuming flat coadding)
        # Total coadding (including diagonal)
        weights_total = np.ones(self.nsplits,dtype=float)/self.nsplits
        spectra_coadd_total = np.einsum('i,ijklmno,j',
                                        weights_total,
                                        spectra,
                                        weights_total)
        # Off-diagonal coadding
        spectra_coadd_xcorr = np.mean(spectra[np.triu_indices(self.nsplits,1)],axis=0)

        # Noise power spectra
        spectra_coadd_noise = spectra_coadd_total - spectra_coadd_xcorr

        # Nulls
        spectra_nulls=np.zeros([self.n_nulls,self.nbands,2,self.nbands,2,self.n_bpws])
        for i_null,(i,j,k,l) in enumerate(self.pairings):
            spectra_nulls[i_null]=spectra[i,k]-spectra[i,l]-spectra[j,k]+spectra[j,l]
            
        # Turn into SACC means
        spectra_coadd_total=spectra_coadd_total.reshape([2*self.nbands,2*self.nbands,self.n_bpws])[np.triu_indices(2*self.nbands)]
        spectra_coadd_xcorr=spectra_coadd_xcorr.reshape([2*self.nbands,2*self.nbands,self.n_bpws])[np.triu_indices(2*self.nbands)]
        spectra_coadd_noise=spectra_coadd_noise.reshape([2*self.nbands,2*self.nbands,self.n_bpws])[np.triu_indices(2*self.nbands)]
        spectra_nulls=spectra_nulls.reshape([-1,self.n_bpws])
        sv_coadd_total=sacc.MeanVec(spectra_coadd_total.flatten())
        sv_coadd_xcorr=sacc.MeanVec(spectra_coadd_xcorr.flatten())
        sv_coadd_noise=sacc.MeanVec(spectra_coadd_noise.flatten())
        sv_nulls=sacc.MeanVec(spectra_nulls.flatten())

        return sv_coadd_total, sv_coadd_xcorr, sv_coadd_noise, sv_nulls

    def run(self):
        # Set things up
        print("Init")
        self.init_params()        

        # Create tracers for all future files
        print("Tracers")
        self.get_tracers(self.s_splits)

        # Create binnings for all future files
        print("Binning")
        self.get_binnings(self)

        # Read data file, coadd and compute nulls
        print("Reading data")
        sv_cd_t, sv_cd_x, sv_cd_n, sv_null=self.parse_splits_sacc_file(self.s_splits,plot_stuff=True)
        
        # Read simulations
        print("Reading simulations")
        sim_cd_t=np.zeros([self.nsims,len(sv_cd_t.vector)])
        sim_cd_x=np.zeros([self.nsims,len(sv_cd_x.vector)])
        sim_cd_n=np.zeros([self.nsims,len(sv_cd_n.vector)])
        sim_null=np.zeros([self.nsims,len(sv_null.vector)])
        for i,fn in enumerate(self.fname_sims):
            s=sacc.SACC.loadFromHDF(fn)
            cd_t,cd_x,cd_n,null=self.parse_splits_sacc_file(s)
            sim_cd_t[i,:]=cd_t.vector
            sim_cd_x[i,:]=cd_x.vector
            sim_cd_n[i,:]=cd_n.vector
            sim_null[i,:]=null.vector

        # Compute covariance
        print("Covariances")
        cov_cd_t=self.get_covariance_from_samples(sim_cd_t,
                                                  covar_type=self.config['data_covar_type'],
                                                  off_diagonal_cut=self.config['data_covar_diag_order'])
        cov_cd_x=self.get_covariance_from_samples(sim_cd_x,
                                                  covar_type=self.config['data_covar_type'],
                                                  off_diagonal_cut=self.config['data_covar_diag_order'])
        cov_cd_n=self.get_covariance_from_samples(sim_cd_n,
                                                  covar_type=self.config['data_covar_type'],
                                                  off_diagonal_cut=self.config['data_covar_diag_order'])
        # There are so many nulls that we'll probably run out of memory
        cov_null=self.get_covariance_from_samples(sim_null,
                                                  covar_type=self.config['nulls_covar_type'],
                                                  off_diagonal_cut=self.config['nulls_covar_diag_order'])

        # Save data
        s_cd_t=self.save_to_sacc(self.get_output("cells_coadded_total"),
                                 self.t_coadd,self.bins_coadd,sv_cd_t,cov=cov_cd_t,
                                 return_sacc=self.config['do_plots'])
        s_cd_x=self.save_to_sacc(self.get_output("cells_coadded"),
                                 self.t_coadd,self.bins_coadd,sv_cd_x,cov=cov_cd_x,
                                 return_sacc=self.config['do_plots'])
        s_cd_n=self.save_to_sacc(self.get_output("cells_noise"),
                                 self.t_coadd,self.bins_coadd,sv_cd_n,cov=cov_cd_n,
                                 return_sacc=self.config['do_plots'])
        s_null=self.save_to_sacc(self.get_output("cells_null"),
                                 self.t_nulls,self.bins_nulls,sv_null,cov=cov_null,
                                 return_sacc=self.config['do_plots'])

        # Plot stuff
        if self.config['do_plots']:
            msk = self.ells<300
            # Nulls
            print("plotting")
            cols={'EE':'r','EB':'g','BE':'y','BB':'b'}
            sorter=s_null.sortTracers()
            xcorrs=np.array(["%d_%d"%(s[0],s[1]) for s in sorter])
            xc_un=np.unique(xcorrs)

            err_null=np.sqrt(s_null.precision.getCovarianceMatrix())
            cls_null=s_null.mean.vector
            for comb in xc_un:
                t1,t2=comb.split('_')
                t1=int(t1)
                t2=int(t2)
                ind_spectra=np.where(xcorrs==comb)[0]
                title="cl_"+self.t_nulls[t1].name+"_x_"+self.t_nulls[t2].name
                print(title)
                plt.figure()
                plt.title(self.t_nulls[t1].name+' x '+self.t_nulls[t2].name)
                for ind in ind_spectra:
                    typ=sorter[ind][2].decode()
                    ndx=sorter[ind][4]
                    plt.errorbar(self.ells[msk], (cls_null[ndx]/err_null[ndx])[msk],
                                 yerr=np.ones(len(ndx))[msk],
                                 fmt=cols[typ]+'-',label=typ)
                plt.xlabel('$\\ell$',fontsize=15)
                plt.ylabel('$C_\\ell/\\sigma_\\ell$',fontsize=15)
                plt.legend()
                self.save_figure(title)
                plt.close()

            s_fid=sacc.SACC.loadFromHDF(self.get_input('cells_fiducial'))
            sorter=s_fid.sortTracers()
            for t1,t2,typ,ells,ndx in sorter:
                typ=typ.decode()
                title='cl_'+s_cd_t.tracers[t1].name+'_'+s_cd_t.tracers[t2].name+'_'+typ
                print(title)
                ct=s_cd_t.mean.vector[ndx]
                cx=s_cd_x.mean.vector[ndx]
                cn=s_cd_n.mean.vector[ndx]
                cf=s_fid.mean.vector[ndx]
                et=np.sqrt(np.diag(s_cd_t.precision.getCovarianceMatrix()))[ndx]
                ex=np.sqrt(np.diag(s_cd_x.precision.getCovarianceMatrix()))[ndx]
                en=np.sqrt(np.diag(s_cd_n.precision.getCovarianceMatrix()))[ndx]
                plt.figure()
                plt.title(s_cd_t.tracers[t1].name+' x '+s_cd_t.tracers[t2].name+' '+typ)
                plt.plot(self.ells[msk], cf[msk],'k-','Fiducial model')
                plt.errorbar(self.ells[msk], ct[msk],yerr=et[msk],fmt='ro-',label='Total coadd')
                plt.errorbar(self.ells[msk],-ct[msk],yerr=et[msk],fmt='rs-')
                plt.errorbar(self.ells[msk], cn[msk],yerr=en[msk],fmt='yo-',label='Noise')
                plt.errorbar(self.ells[msk],-cn[msk],yerr=en[msk],fmt='ys-')
                plt.errorbar(self.ells[msk], cx[msk],yerr=ex[msk],fmt='bo-',label='Cross-coadd')
                plt.errorbar(self.ells[msk],-cx[msk],yerr=ex[msk],fmt='bs-')
                plt.yscale('log')
                plt.xlabel('$\\ell$',fontsize=15)
                plt.ylabel('$C_\\ell$',fontsize=15)
                plt.legend()
                self.save_figure(title)
                plt.close()

if __name__ == '__main_':
    cls = PipelineStage.main()
