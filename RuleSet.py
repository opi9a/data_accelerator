# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from r_funcs import *
from projection_funcs import get_ix_slice, slicify, variablise
import inspect


class RuleSet:
    def __init__(self, parent_df, name, string_slice=None, index_slice=None,
                func_str="", func=None, f_args={}, join_output=True):
        self.name = name
        self.parent_df = parent_df
        
        # initialise the index slice to the names in the parent index
        if index_slice is None:
            self.index_slice = {i:None for i in parent_df.index.names}
        else:
            self.index_slice = index_slice

        # string slice holds slicing info in format used by input form
        self.string_slice = string_slice

        self.func = func
        self.func_str = func_str
        self.f_args = f_args
        self.past = None
        self.fut = None
        self.join_output=join_output
        self.joined = None
        self.summed = None
        self.out_fig = ""

       
    def set_slice(self, input_slice):
        '''Set the index_slice dictionary.

        Takes a dictionary input, and sets matching elements in self.index_slice
        to equal the values in the input_slice'''

        for k in input_slice:
            if k in self.index_slice:
                self.index_slice[k] = input_slice[k]
            else:
                print('{', k, ':', input_slice[k], '} not in index')


    def slicify_string(self):
        for i in self.index_slice:
             self.index_slice[i] = slicify(self.string_slice[i])


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
        if self.func == None:
            print("no function assigned")
            return
        slice= get_ix_slice(self.parent_df, self.index_slice)
        self.past = self.parent_df.loc[slice,:]
        
        self.fut = self.func(self.past, n_pers, **self.f_args)

        if self.join_output:
            try:
                self.joined = self.past.join(self.fut)
                self.summed = pd.DataFrame(self.joined.sum(), columns=[self.name])
            except:
                self.joined = self.summed = pd.DataFrame(self.past.sum().append(self.fut).sum(axis=1), 
                                                columns=[self.name])

         
        
    def __repr__(self):
        outlist = []
        self._info = {"name": self.name,
             "string_slice": self.string_slice,
             "index_slice": self.index_slice,
             "func": self.func,
             "f_args keys": list(self.f_args.keys()),
             "f_args": self.f_args,
             "past type": type(self.past),
             "fut type": type(self.fut),
             "joined type": type(self.joined),
             "summed type": type(self.summed),
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
    
    
    