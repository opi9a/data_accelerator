# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from r_funcs import *
import projection_funcs
import inspect


class RuleSet:
    def __init__(self, parent_df, name, index_slice={}, 
                 func=None, f_args={}, join_output=True):
        self.name = name
        self.parent_df = parent_df
        self.index_dims = self.parent_df.index.names
        self.index_slice = index_slice
        self.func = func
        self.f_args = f_args
        self.past = None
        self.fut = None
        self.join_output=join_output
        self.joined = None

       
    def set_slice(self, a_slice):
        '''Set the index_slice dictionary'''

        wrong_keys = [k for k in a_slice.keys() 
                        if k not in self.index_dims]
        if wrong_keys:
            print("These keys aren't in the index: ", wrong_keys)

        else: self.index_slice = a_sli


    def set_args(self, a_args):
        '''Set the f_args dictionary - passed to self.func()'''

        if self.func==None:
            print("no function yet")
            return

        wrong_keys = [k for k in a_args.keys() 
                        if k not in inspect.getfullargspec(self.func)[4]]

        if wrong_keys:
            print("Args aren't parameters of ", self.func.__name__,": ", wrong_keys)

        else: self.f_args = a_args



    def get_params(self):
        #todo: parse out in best way
        if self.func==None:
            print("no function yet")

        else:
            return inspect.getfullargspec(self.func)[4]

    

    def xtrap(self, n_pers):
        self.past = self.parent_df.loc[projection_funcs.get_ix_slice(
                        self.parent_df, **self.index_slice),:]
        self.fut = self.func(self.past, n_pers, **self.f_args)
        
        if self.join_output:
            try:
                self.joined = self.past.join(self.fut)
            except:
                self.joined = pd.DataFrame(pd.concat([self.past.sum(),self.fut.sum()]), 
                                            columns=['joined']).T

        
    def __repr__(self):
        outlist = []
        self._info = {"name": self.name,
             "index_dims": self.index_dims,
             "index_slice": self.index_slice,
             "func": self.func,
             "f_args keys": list(self.f_args.keys()),
             "f_args": self.f_args,
             "past type": type(self.past),
             "fut type": type(self.fut),
             "join_output": self.join_output}
        for key in self._info:
            temp_string = key.ljust(27) + str(self._info[key]).rjust(10)
            outlist.append(temp_string)
        return "\n".join(outlist)
    


    
if __name__ == "__main__":
    df = pd.read_pickle('c:/Users/groberta/Work/data_accelerator/spend_data_proc/dfs/main_unstacked_17AUG.pkl')
    prof1 = np.genfromtxt('c:/Users/groberta/Work/data_accelerator/profiles/prof1.csv')
    cutoff = pd.Period('3-2014', freq='M')
    
    test=2
    
    if test == 1:    
        test_r = RuleSet(df, 't1')
        test_r.index_slice = {'biol':True}
        test_r.func = funcs.r_profile
        test_r.f_args = {'profile':prof1, 'gen_mult':0.5}
        print(test_r)
        test_r.xtrap(120)

    if test == 2:    
        test_r = RuleSet(df, 't2')
        test_r.index_slice = {'start_date':slice(cutoff,None,None)}
        test_r.func = funcs.r_fut
        test_r.f_args = {'profile':prof1, 'cutoff_date':cutoff, 
                            'coh_growth':0, 'term_growth':0, 'name':'fut'}
        print(test_r)
        test_r.xtrap(120)
    
    
    