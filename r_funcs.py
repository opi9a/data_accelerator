# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import projection_funcs

##_________________________________________________________________________##

'''Suite of functions with common API which can be passed to a RuleSet object

TODO: document the API etc
'''

##_________________________________________________________________________##


def r_profile(df, n_pers, 
              *, profile, gen_mult, debug=False):
    '''Applies a profile, with a variable multiplier for patent expiry
    
    args:   data (df or series - future) - which includes index data eg launch year
            n_pers
            r_args - to include the profile itself and the gen_mult
            
    return: df (or series - future)
    '''
    out=[]

    for row in df.itertuples():
        #note the structure of each row:  [0] is the index (a tuple), [1:] is the data
        launch_date = row[0][df.index.names.index('start_date')] # gets the index number of 'start_date' in the df.index list
        last_date = df.columns[-1]
        last_spend = row[-1]
        start_x = last_date - launch_date
        basic = profile[start_x: start_x+n_pers]
        scaling_f = last_spend / profile[start_x-1]
        y = basic * scaling_f
        out.append(y)
    
    if debug:
        pad1 = 20
        print("launch_date".ljust(pad1), launch_date, 
              "\nlast_date".ljust(pad1), last_date, 
              "\nn_pers".ljust(pad1), n_pers, 
              "\nlast_spend".ljust(pad1), last_spend, 
              "\nstart_x".ljust(pad1), start_x, 
              "\nscaling_f".ljust(pad1), scaling_f)
    # Build the df index and columns
    ind = df.index
    cols = pd.PeriodIndex(start=last_date+1, periods=n_pers, freq='M')
    
    return pd.DataFrame(out, index=df.index, columns=cols)

##_________________________________________________________________________##

def r_terminal(df, n_pers, *, term_rate_pa, debug=False):
    '''take average of last 3 monthly values and extrapolate at terminal rate
    
    default args:   data_in, n_pers
    
    r_args:         term_rate_pa     - terminal rate, annual (positive)
    
    return:         dataframe
    
    '''
    # make an initial np array with growth, based on index-1 = 1

    term_rate_mo = term_rate_pa / 12
    x = np.array([(1+term_rate_mo)**np.arange(1,n_pers+1)]*len(df))
    ave_last = df.iloc[:,-3:].sum(axis=1)/3
    x=x*ave_last[:,None]
    
    # Build the df index and columns
    ind = df.index
    last_date = df.columns[-1]
    cols = pd.PeriodIndex(start=last_date+1, periods=n_pers, freq='M')
    
    return pd.DataFrame(x, index=ind, columns=cols)


##_________________________________________________________________________##

def r_fut(df, n_pers, *, profile, cutoff_date,
          coh_growth, term_growth, name='future', debug=False):
    # work out what l_start and l_stop should be, and how passed to get_forecast()

    # sum the df, as we don't care about individual products.  NB it's now a Series
    df=df.sum()
    last_date = df.index[-1]

    l_start=0
    l_stop=(last_date - cutoff_date)+n_pers
    proj_start=(last_date - cutoff_date)
    proj_stop=(last_date - cutoff_date)+n_pers
    
    if debug: 
        pad=20
        print("n_pers".ljust(pad), n_pers,"\nlast_date".ljust(pad), last_date,"\ncutoff_date".ljust(pad), cutoff_date,"\nl_start".ljust(pad), l_start,"\nl_stop".ljust(pad), l_stop,"\nproj_start".ljust(pad), proj_start,"\nproj_stop".ljust(pad), proj_stop)
    
    # note this gets a projection for one period behind the required, to allow scaling.  This must be sliced off later.
    fut = projection_funcs.get_forecast(profile, l_start, l_stop, coh_growth,term_growth,1, proj_start-1, proj_stop, name=name)
    if debug: print(fut)
        
    # now get scaling factor. Take cumulative sum for last period in slice passed
    # (also should have available the actual period from the slice - do later)
    last_sum = df[-1]

    # to scale, want the period just before the fut forecast to equal last_sum.  Deliberately overlap, and then snip off first period?
    scaler=last_sum/fut.iloc[0]
    
    if debug: print(last_sum, scaler)
    
    out = (fut*scaler)[1:]
    
    out.index=pd.PeriodIndex(start=last_date+1, periods=n_pers, freq='M')

    return pd.DataFrame(out)