from bbpipe import PipelineStage
from .types import TextFile, SACCFile, DirFile, HTMLFile, NpzFile
import sacc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import dominate as dom
import dominate.tags as dtg
import os

class BBPlotter(PipelineStage):
    name="BBPlotter"
    inputs=[('cells_coadded_total',SACCFile), ('cells_coadded',SACCFile),
            ('cells_noise',SACCFile), ('cells_null',SACCFile), 
            ('cells_fiducial',SACCFile), ('param_chains',NpzFile)]
    outputs=[('plots',DirFile),('plots_page',HTMLFile)]
    config_options={'lmax_plot':300}

    def create_page(self):
        # Open plots directory
        if not os.path.isdir(self.get_output('plots')):
            os.mkdir(self.get_output('plots'))

        # Create HTML page
        self.doc = dom.document(title='BBPipe plots page')
        with self.doc.head:
            dtg.link(rel='stylesheet', href='style.css')
            dtg.script(type='text/javascript', src='script.js')
        with self.doc:
            dtg.h1("Pipeline outputs")
            dtg.h2("Contents:",id='contents')
            lst=dtg.ul()
            lst+=dtg.li(dtg.a('Bandpasses',href='#bandpasses'))
            lst+=dtg.li(dtg.a('Coadded power spectra',href='#coadded'))
            lst+=dtg.li(dtg.a('Null tests',href='#nulls'))
            lst+=dtg.li(dtg.a('Likelihood',href='#like'))

    def add_bandpasses(self):
        with self.doc:
            dtg.h2("Bandpasses",id='bandpasses')
            lst=dtg.ul()
            # Overall plot
            title='Bandpasses summary'
            fname=self.get_output('plots')+'/bpass_summary.png'
            plt.figure()
            plt.title(title,fontsize=14)
            for t in self.s_fid.tracers:
                n=t.name[2:-1]
                nu_mean=np.sum(t.Nz*t.z**3*t.extra_cols['dnu'])/np.sum(t.Nz*t.z**2*t.extra_cols['dnu'])
                plt.plot(t.z,t.Nz/np.amax(t.Nz),label=n+', $\\langle\\nu\\rangle=%.1lf\\,{\\rm GHz}$'%nu_mean)
            plt.xlabel('$\\nu\\,[{\\rm GHz}]$',fontsize=14)
            plt.ylabel('Transmission',fontsize=14)
            plt.ylim([0.,1.3])
            plt.legend(frameon=0,ncol=2,labelspacing=0.1,loc='upper left')
            plt.xscale('log')
            plt.savefig(fname,bbox_inches='tight')
            plt.close()
            lst+=dtg.li(dtg.a(title,href=fname))
            
            for t in self.s_fid.tracers:
                n=t.name[2:-1]
                title='Bandpass '+n
                fname=self.get_output('plots')+'/bpass_'+n+'.png'
                plt.figure()
                plt.title(title,fontsize=14)
                plt.plot(t.z,t.Nz/np.amax(t.Nz))
                plt.xlabel('$\\nu\\,[{\\rm GHz}]$',fontsize=14)
                plt.ylabel('Transmission',fontsize=14)
                plt.ylim([0.,1.05])
                plt.savefig(fname,bbox_inches='tight')
                plt.close()
                lst+=dtg.li(dtg.a(title,href=fname))
            dtg.div(dtg.a('Back to TOC',href='#contents'))

    def add_coadded(self):
        cls_f=self.s_fid.mean.vector
        cls_t=self.s_cd_t.mean.vector
        els_t=np.sqrt(np.diag(self.s_cd_t.precision.getCovarianceMatrix()))
        cls_x=self.s_cd_x.mean.vector
        els_x=np.sqrt(np.diag(self.s_cd_x.precision.getCovarianceMatrix()))
        cls_n=self.s_cd_n.mean.vector
        els_n=np.sqrt(np.diag(self.s_cd_n.precision.getCovarianceMatrix()))

        with self.doc:
            dtg.h2("Coadded power spectra",id='coadded')
            lst=dtg.ul()
            sorter=self.s_fid.sortTracers()
            # Loop over all possible power spectra
            for t1,t2,typ,ells,ndx in sorter:
                typ=typ.decode()
                # Plot title
                title =self.s_cd_t.tracers[t1].name[2:-1]
                title+=" x "
                title+=self.s_cd_t.tracers[t2].name[2:-1]
                title+=" "+typ
                # Plot file
                fname =self.get_output('plots')+'/cls_'
                fname+=self.s_cd_t.tracers[t1].name[2:-1]
                fname+="_x_"
                fname+=self.s_cd_t.tracers[t2].name[2:-1]
                fname+="_"+typ+".png"
                print(fname)
                cf=cls_f[ndx][self.msk]
                ct=cls_t[ndx][self.msk]
                et=els_t[ndx][self.msk]
                cn=cls_n[ndx][self.msk]
                en=els_n[ndx][self.msk]
                cx=cls_x[ndx][self.msk]
                ex=els_x[ndx][self.msk]
                # For each combination, plot signal, noise, total and fiducial model
                plt.figure()
                plt.title(title,fontsize=14)
                plt.plot(self.ells[self.msk],cf,'k-',label='Fiducial model')
                plt.errorbar(self.ells[self.msk], ct,yerr=et,fmt='ro',
                             label='Total coadd')
                eb=plt.errorbar(self.ells[self.msk]+1.,-ct,yerr=et,fmt='ro',mfc='white')
                eb[-1][0].set_linestyle('--')

                plt.errorbar(self.ells[self.msk], cn,yerr=et,fmt='yo',
                             label='Noise')
                eb=plt.errorbar(self.ells[self.msk]+1.,-cn,yerr=et,fmt='yo',mfc='white')
                eb[-1][0].set_linestyle('--')

                plt.errorbar(self.ells[self.msk], cx,yerr=et,fmt='bo',
                             label='Cross-coadd')
                eb=plt.errorbar(self.ells[self.msk]+1.,-cx,yerr=et,fmt='bo',mfc='white')
                eb[-1][0].set_linestyle('--')
                plt.yscale('log')
                plt.xlabel('$\\ell$',fontsize=15)
                plt.ylabel('$C_\\ell$',fontsize=15)
                plt.legend()
                plt.savefig(fname,bbox_inches='tight')
                plt.close()
                lst+=dtg.li(dtg.a(title,href=fname))

            dtg.div(dtg.a('Back to TOC',href='#contents'))
                
    def add_nulls(self):
        with self.doc:
            dtg.h2("Null tests",id='nulls')
            lst=dtg.ul()

            sorter=self.s_null.sortTracers()
            # All cross-correlations
            xcorrs=np.array(["%d_%d"%(s[0],s[1]) for s in sorter])
            # Unique cross-correlations
            xc_un=np.unique(xcorrs)

            cls_null=self.s_null.mean.vector
            err_null=np.sqrt(self.s_null.precision.getCovarianceMatrix())
            # Loop over unique correlations
            for comb in xc_un:
                t1,t2=comb.split('_')
                t1=int(t1)
                t2=int(t2)
                # Find all power spectra for this pair of tracers
                ind_spectra=np.where(xcorrs==comb)[0]
                # Plot title
                title =self.s_null.tracers[t1].name[2:-1]
                title+=" x "
                title+=self.s_null.tracers[t2].name[2:-1]
                # Plot file
                fname =self.get_output('plots')+'/cls_null_'
                fname+=self.s_null.tracers[t1].name[2:-1]
                fname+="_x_"
                fname+=self.s_null.tracers[t2].name[2:-1]
                fname+=".png"
                print(fname)

                # Plot all power spectra
                plt.figure()
                plt.title(title,fontsize=15)
                for ind in ind_spectra:
                    typ=sorter[ind][2].decode()
                    ndx=sorter[ind][4]
                    plt.errorbar(self.ells[self.msk],
                                 (cls_null[ndx]/err_null[ndx])[self.msk],
                                 yerr=np.ones(len(ndx))[self.msk],
                                 fmt=self.cols_typ[typ]+'-',label=typ)
                plt.xlabel('$\\ell$',fontsize=15)
                plt.ylabel('$C_\\ell/\\sigma_\\ell$',fontsize=15)
                plt.legend()
                plt.savefig(fname,bbox_index='tight')
                plt.close()
                lst+=dtg.li(dtg.a(title,href=fname))

            dtg.div(dtg.a('Back to TOC',href='#contents'))

    def add_contours(self):
        from getdist import MCSamples
        from getdist import plots as gplots

        with self.doc:
            dtg.h2("Likelihood",id='like')
            lst=dtg.ul()

            # Labels and true values
            labdir={'A_lens':'A_{\\rm lens}',
                    'r_tensor':'r',
                    'beta_d':'\\beta_d',
                    'epsilon_ds':'\\epsilon_{ds}',
                    'alpha_d_bb':'\\alpha_d',
                    'amp_d_bb':'A_d',
                    'beta_s':'\\beta_s',
                    'alpha_s_bb':'\\alpha_s',
                    'amp_s_bb':'A_s'}
            # TODO: we need to build this from the priors, I think.
            truth={'A_lens':1.,
                   'r_tensor':0.,
                   'beta_d':1.59,
                   'epsilon_ds':0.,
                   'alpha_d_bb':-0.42,
                   'amp_d_bb':5.,
                   'beta_s':-3.,
                   'alpha_s_bb':-0.6,
                   'amp_s_bb':2.}

            # Select only parameters for which we have labels
            names_common=list(set(list(self.chain['names'])) & truth.keys())
            msk_common=np.array([n in names_common for n in self.chain['names']])
            npar=len(names_common)
            nwalk,nsamp,npar_chain=self.chain['chain'].shape
            chain=self.chain['chain'][:,nsamp//4:,:].reshape([-1,npar_chain])[:,msk_common]
            names_common=np.array(self.chain['names'])[msk_common]

            # Getdist
            samples=MCSamples(samples=chain,
                              names=names_common,
                              labels=[labdir[n] for n in names_common])
            g = gplots.getSubplotPlotter()
            g.triangle_plot([samples], filled=True)
            for i,n in enumerate(names_common):
                v=truth[n]
                g.subplots[i,i].plot([v,v],[0,1],'r-')
                for j in range(i+1,npar):
                    u=truth[names_common[j]]
                    g.subplots[j,i].plot([v],[u],'ro')

            # Save
            fname=self.get_output('plots')+'/triangle.png'
            g.export(fname)
            lst+=dtg.li(dtg.a("Likelihood contours",href=fname))

            dtg.div(dtg.a('Back to TOC',href='#contents'))

    def write_page(self):
        with open(self.get_output('plots_page'),'w') as f:
            f.write(self.doc.render())

    def read_inputs(self):
        print("Reading inputs")
        # Power spectra
        self.s_fid=sacc.SACC.loadFromHDF(self.get_input('cells_fiducial'),read_windows=False)
        self.s_cd_t=sacc.SACC.loadFromHDF(self.get_input('cells_coadded_total'),read_windows=False)
        self.s_cd_x=sacc.SACC.loadFromHDF(self.get_input('cells_coadded'),read_windows=False)
        self.s_cd_n=sacc.SACC.loadFromHDF(self.get_input('cells_noise'),read_windows=False)
        self.s_null=sacc.SACC.loadFromHDF(self.get_input('cells_null'),read_windows=False)
        # Chains
        self.chain=np.load(self.get_input('param_chains'))

        self.ells=self.s_fid.sortTracers()[0][3]
        self.msk=self.ells<self.config['lmax_plot']
        self.cols_typ={'EE':'r','EB':'g','BE':'y','BB':'b'}

    def run(self):
        self.read_inputs()
        self.create_page()
        self.add_bandpasses()
        self.add_coadded()
        self.add_nulls()
        self.add_contours()
        self.write_page()

if __name__ == '__main_':
    cls = PipelineStage.main()
