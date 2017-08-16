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
        
        self._info = {"name": self.name,
             "index_dims": self.index_dims,
             "index_selection": self.index_selection,
             "r_func": self.r_func,
             "pr_args": max(self.r_args),
             "join_output": self.join_output}
        
    def get_arg_keys(self):
        #todo: parse out in best way
        return inspect.getfullargspec(self.r_func)[4]
    
    def xtrap(self, n_pers):
        self.past = self.parent_df.loc[projection_funcs.get_ix_slice(
                        self.parent_df, **self.index_selection),:]
        self.fut = self.r_func(self.past, n_pers, **self.r_args)
        if self.join_output:
            return self.past.sum().join(self.fut)
        else:
            return self.fut.sum()
        
    def __repr__(self):
        outlist = []
        for key in self._info:
            temp_string = key.ljust(27) + str(self._info[key]).rjust(10)
            outlist.append(temp_string)
        return "\n".join(outlist)