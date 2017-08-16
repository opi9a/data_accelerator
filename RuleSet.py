# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import r_funcs
import projection_funcs
import inspect

class RuleSet:
    def __init__(self, parent_df, name, 
                 index_selection={}, r_func=None, r_args={}):
        self.name = name
        self.parent_df = parent_df
        self.index_dims = self.parent_df.index.names
        self.index_selection = index_selection
        self.r_func = r_func
        self.r_args = r_args
        self.past = None
        self.fut = None
        self.all = None
        
    def get_arg_keys(self):
        #todo: parse out in best way
        return inspect.getfullargspec(self.r_func)[4]
    
    def xtrap(self, n_pers):
        self.past = self.parent_df.loc[projection_funcs.get_ix_slice(
                        self.parent_df, **self.index_selection),:]
        self.fut = self.r_func(self.past, n_pers, **self.r_args)
        self.all = self.past.join(self.fut)
        return self.all