from bbpipe import PipelineStage
from types_mine import TextFile, SACCFile, DirFile, HTMLFile, NpzFile
#import sacc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import dominate as dom
import dominate.tags as dtg
import os

class BB_plotter(PipelineStage):
    name="BB_plotter"
    inputs = [('param_chains', NpzFile)]
    outputs=[('plots',DirFile),('plots_page',HTMLFile)]
    config_options={'lmax_plot':1000}

    def read_inputs(self):
        print("Reading inputs")
        # Chains
        self.chain=np.load(self.get_input('param_chains'))
        self.cols_typ={'EE':'r','EB':'g','BE':'y','BB':'b'}

    def create_page(self):
        # Open plots directory
        if not os.path.isdir(self.get_output('plots')):
            print('HERE')
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
    
    def run(self):
        self.read_inputs()
        self.create_page()
        #self.add_bandpasses()
        #self.add_coadded()
        #self.add_nulls()
        self.add_contours()
        self.write_page()

if __name__ == '__main_':
    cls = PipelineStage.main()