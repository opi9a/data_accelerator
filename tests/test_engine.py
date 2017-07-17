# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 14:15:09 2017

To run these tests either just run from this file or use following command 
from main directory:
    
    $ python -m pytest .
    
Or, if make is installed:
    
    $ make test
    
should work

"""

from spend_engine_funcs import *

def test_spend_engine():
    test_profile2 = np.array([1,2,5,10,11,12,12,1])
    assert sum(get_forecast(test_profile2, 0,10,0.1,-0.1,1,0,15)) == 853.6565405610005
