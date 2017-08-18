# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import r_funcs
import projection_funcs
import inspect


class RuleSet:
    def __init__(self, parent_df, name, index_selection={}, 
                 r_func=None, r_args={}, join_output=True):
        self.name = name
        self.parent_df = parent_df
        self.index_dims = self.parent_df.index.names
        self.index_selection = index_selection
        self.r_func = r_func
        self.r_args = r_args
        self.past = None
        self.fut = None
        self.join_output=join_output
        self.joined = None
       
    def get_arg_keys(self):
        #todo: parse out in best way
        return inspect.getfullargspec(self.r_func)[4]
    
    def xtrap(self, n_pers):
        self.past = self.parent_df.loc[projection_funcs.get_ix_slice(
                        self.parent_df, **self.index_selection),:]
        self.fut = self.r_func(self.past, n_pers, **self.r_args)
        
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
             "index_selection": self.index_selection,
             "r_func": self.r_func,
             "r_args keys": list(self.r_args.keys()),
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
        test_r.index_selection = {'biol':True}
        test_r.r_func = r_funcs.r_profile
        test_r.r_args = {'profile':prof1, 'gen_mult':0.5}
        print(test_r)
        test_r.xtrap(120)

    if test == 2:    
        test_r = RuleSet(df, 't2')
        test_r.index_selection = {'start_date':slice(cutoff,None,None)}
        test_r.r_func = r_funcs.r_fut
        test_r.r_args = {'profile':prof1, 'cutoff_date':cutoff, 
                            'coh_growth':0, 'term_growth':0, 'name':'fut'}
        print(test_r)
        test_r.xtrap(120)
    
    
    