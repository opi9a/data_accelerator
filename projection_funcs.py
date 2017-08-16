# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 14:50:32 2017

@author: GRoberta
"""




def get_ix_slice(df, **kwargs):
    '''make a pd.IndexSlice
    args:   - a dataframe (with named index)
            - index names:value pairs to be sliced (in any order)
            
    returns a pd.IndexSlice with the desired spec
            
    eg, if index contains the boolean 'is_cool' and ints 'year'
       'is_cool = True' will generate a slice where 'is_cool'  is set to True
       'year = slice(2006, 2009, None)' will select years 2006 to 2009 
       'year = slice(2006, None, None)' will select years 2006 onward 
       'year = [2008, 2012]' will select just those two years
       
    simply print the returned output to see what's going on
    '''
    return tuple((kwargs.get(name, slice(None,None,None)))
                     for name in df.index.names)
    
 ###_________________________________________________________________________###

   


###_________________________________________________________________________###




###_________________________________________________________________________###



###_________________________________________________________________________###

