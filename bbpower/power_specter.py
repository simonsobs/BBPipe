from bbpipe import PipelineStage
from .types import FitsFile,YamlFile,DummyFile
from sacc.sacc import SACC

class BBPowerSpecter(PipelineStage):
    """
    Template for a power spectrum stage
    """
    name="BBPowerSpecter"
    inputs=[('splits_info',YamlFile),('window_function',FitsFile),('filtered_maps',FitsFile)]
    outputs=[('cells_coadded',SACC),('cells_noise',SACC),('cells_splits',SACC)]
    config_options={'bpw_edges':[90,110,130],
                    'beam_correct':True,
                    'pcl_code':'NaMaster',
                    'pure_b':True}

    def init_all(self):
        self.n_bpw=len(self.config['bpw_edges'])
        if self.config['pcl_code']!='NaMaster':
            raise NotImplementedError("Only power spectrum method available is \"NaMaster\"")
        self.process_splits_info()
        self.read_window_functions()
        raise NotImplementedError("Not implemented yet")

    def read_window_functions(self):
        self.window=None
        raise NotImplementedError("Not implemented yet")
        
    def process_splits_info(self):
        #TODO: parse splits file
        self.bands=None
        self.splits=None
        raise NotImplementedError("Not implemented yet")

    def get_fields(self):
        for b in bands:
            window=self.window[b]
            for s in self.splits[b]:
                s.field=nmt.NmtField(self.window,[s1.map_q,s1.map_u],purify_b=self.config['pure_b'])
        raise NotImplementedError("Not tested yet")
    
    def get_power_spectrum_from_splits(self,band1,band2=None):
        if band2 is None:
            band2=band1

        is_auto= band2==band1

        n_splits1=len(self.splits[band1])
        n_splits2=len(self.splits[band2])

        c_ells=np.zeros([n_splits1,n_splits2,4,self.n_bpw])
        for i1,s1 in enumerate(self.splits[band1]):
            for i2,s2 in enumerate(self.splits[band2]):
                if is_auto and i2<i1:
                    c_ells[i1,i2,:,:]=c_ells[i2,i1,:,:]
                    continue
                c_ells[i1,i2,:,:]=self.wsp.decouple_cell(nmt.compute_coupled_cell(s1.field,s2.field))

        raise NotImplementedError("Not tested yet")
        return c_ells

    def get_coadded_spectra(self,x_splits):
        raise NotImplementedError("Not implemented yet")

    def write_output(self):
        raise NotImplementedError("Not implemented yet")

    def compute_covariance_matrix(self):
        raise NotImplementedError("Not implemented yet")
        
    def run(self) :
        #Process inputs
        self.init_all()

        #Compute power spectra
        self.c_ells_all={}
        self.c_ells_coadd={}
        for i2,b1 in enumerate(self.bands):
            self.c_ells_all[b1]={}
            self.c_ells_coadd[b1]={}
            for i2,b2 in enumerate(self.bands):
                if i2<i1:
                    self.c_ells_all[b1][b2]=c_ells_all[b2][b1]
                    self.c_ells_coadd[b1][b2]=c_ells_coadd[b2][b1]
                else:
                    self.c_ells_all[b1][b2]=self.get_power_spectrum_from_splits(b1,band2=b2)
                    self.c_ells_coadd[b1][b2]=self.get_coadded_spectra(self,c_ells_all[b1][b2])

        #Compute covariance
        self.compute_covariance_matrix()

        #Write output
        self.write_output()

if __name__ == '__main__':
    cls = PipelineStage.main()
