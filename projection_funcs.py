# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 14:50:32 2017

@author: GRoberta
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import r_funcs as rf





###_________________________________________________________________________###


def plot_projs(rs_in, num_plots, out_path="", ma_interval=12):
    '''
    For an input ruleset, generates informative plots of individual products
    '''
       
    # get the various datasets
    past_df = rs_in.past
    fut_df = rs_in.fut
    joined_df = rs_in.joined
    
    # if there are relevant arguments to infer_launch() and elsewhere, get them.  
    # Note these have similar but not same names :/
    il_args = {}
    if 'ma_interval' in rs_in.f_args: il_args['_ma_interval'] = rs_in.f_args['ma_interval']
    if 'streak_len_thresh' in rs_in.f_args: il_args['streak_len_threshold'] = rs_in.f_args['streak_len_thresh']
    if 'delta_thresh' in rs_in.f_args: il_args['delta_threshold'] = rs_in.f_args['delta_thresh']
    ma_interval = il_args.get('_ma_interval', ma_interval)
  
    # make sure don't go beyond available products
    num_plots = min(num_plots, len(past_df))  
    
    # need these pads to fill out series for plotting later
    past_pad = len(joined_df.T) - len(past_df.T)
    fut_pad = len(joined_df.T) - len(fut_df.T)
    
    # make the selection, ordering by max sales *ever* (not just in past)
    subset = fut_df.max(axis=1).sort_values(ascending=False).head(num_plots).index.get_level_values(level=0)
    past_df = past_df.loc[list(subset),:]
    fut_df = past_df.loc[list(subset),:]
    
    fig, axs = plt.subplots(num_plots, 1, figsize=(10,5*num_plots))
    ind = joined_df.columns.to_timestamp()

    # this is useful for efficiently getting max_sales
    index_df = pd.DataFrame([*past_df.index], columns = past_df.index.names).set_index(keys='PxMolecule')

    for i,p in enumerate(subset):

        max_sales = index_df.loc[p,'max_sales']
        uptake_out = rf.infer_launch(past_df.loc[p].values, max_sales, return_dict=True, _debug=False, **il_args)

        # basic plot of joined (i.e. past plus future)
        axs[i].plot(ind, joined_df.loc[p].T, label = 'actual and future spend')
        
        # moving average
        ma = mov_ave(past_df.loc[p], ma_interval)
        axs[i].plot(ind, np.append(ma,[np.nan]*past_pad), label= 'moving average, interval: ' + str(ma_interval))
        
        # lagged moving average - useful for identifying patterns
        lagged_ma = ma - np.insert(ma, 0, np.zeros(ma_interval))[:-ma_interval]
        axs[i].plot(ind, np.append(lagged_ma,[np.nan]*past_pad), label = 'moving diff in moving average')
           
        # show detected streaks as shaded regions
        if uptake_out['uptake_detected']:
            for s in uptake_out['streaks']:
                up_end = uptake_out['streaks'][s]['end_per']
                up_start = up_end - uptake_out['streaks'][s]['length'] - int(ma_interval/2)

                axs[i].axvspan(ind[up_start], ind[up_end], alpha=0.1)
        
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['bottom'].set_position('zero')
        axs[i].set_title(p)
        if (i%5 == 0): axs[i].legend()
    if not os.path.exists(out_path+'/plots'):
        os.makedirs(out_path+'/plots')    
    fig.savefig(os.path.join(out_path +'/plots/' + rs_in.name + '_plots.png'))



###_________________________________________________________________________###

def mov_ave(in_arr, window):
    '''Parameters:  
        
            in_arr: an input array (numpy, or anything that can be coerced by np.array())
            window: the window over which to make the moving average


        Return:

            array of same length as in_arr, with mov ave
    '''
    
    # first coerce to numpy array 
    in_arr = np.array(in_arr)    

    # now turn nans to zero
    in_arr[np.isnan(in_arr)]=0

    a = np.cumsum(in_arr) # total cumulative sum
    b=(np.cumsum(in_arr)[:-window]) # shifted forward, overlap truncated
    c = np.insert(b,0,np.zeros(window))  # start filled to get to line up
    return(a-c)/window


###_________________________________________________________________________###

def variablise(string):
    '''Turns an input string into a variable, if it can.

    Variables tried: bool, int, float, pd.Period
    '''
    if string is None:
        print('variablising but None for string')
        return None

    if string.strip().lower() == 'true':
        return True
    elif string.strip().lower() == 'false':
        return False
    
    else:
        try: 
            return int(string)
        except:
            try:
                return float(string)
            except:
                try:
                    return pd.Period(string)
                except: return string.strip()


 ###_________________________________________________________________________###


def slicify(in_string):
    '''
    Processes index_slice input strings from web form into variables useable by 
    get_ix_slice. 
    '''
    print('calling slicify on ', in_string)

    if in_string == "" or in_string == None:
        return slice(None, None, None)

    # see if string can be variablised (is a bool, float, int or pd.Period)
    v = variablise(in_string)
    if not isinstance(v, str):
        print("..variablised as ", type(v))
        return v
    
    # now deal with list-likes
    if in_string.strip()[0] == "[" and in_string.strip()[-1] == "]":
        out_list = [variablise(i) for i in in_string.strip()[1:-1].split(',')]
        print("..listified as ", out_list)
        return out_list
    
    # finally deal with slices
    if in_string.strip().startswith('slice:'):
        print('looks like a slice.. ', end = "")
        slice_out = in_string[6:].strip().split(' ')
        
        if slice_out[0] == 'to':
            print(' with to only')
            return slice(None, variablise(slice_out[1]),None)

        if slice_out[-1] == 'to':
            print(' with from and to')
            return slice(variablise(slice_out[0]),None,None)

        if slice_out[1] == 'to' and len(slice_out)==3:
            print(' with from only')
            return slice(variablise(slice_out[0]),variablise(slice_out[2]),None)
        
        else:
            print('could not slicify slice.  Returning nothing from slicify()', in_string)

    else:
        print('could not slicify string, returning unprocessed ', in_string)
        return in_string



###_________________________________________________________________________###




def get_ix_slice(df, in_dict):
    '''make a pd.IndexSlice
    args:   - a dataframe (with named index)
            - dict of index names:value pairs to be sliced (in any order)
            
    returns a pd.IndexSlice with the desired spec
            
    eg, if index contains the boolean 'is_cool' and ints 'year'
       'is_cool = True' will generate a slice where 'is_cool'  is set to True
       'year = slice(2006, 2009, None)' will select years 2006 to 2009 
       'year = slice(2006, None, None)' will select years 2006 onward 
       'year = [2008, 2012]' will select just those two years
       
    simply print the returned output to see what's going on

    Can pass output directly to iloc.  Eg

        ixs = pf.get_ix_slice(df_main, dict(is_biol=False, launch_year=[2015,2016]))
        df_main.loc[ixs,:]

    '''
    # first turn any None entries of the input into slices
    for i in in_dict:
        if in_dict[i] is '' or in_dict[i] is None:
            in_dict[i] = slice(None, None, None)
            

    return tuple((in_dict.get(name, slice(None,None,None)))
                     for name in df.index.names)
    
 ###_________________________________________________________________________###

   
# Core spend per period function
# Given input profile, timings etc, returns the spend on each cohort.  
# It's a generator, intended to be consumed as `sum(spender(...))`.

def spender(spend_per, profile, launch_pers, coh_growth, term_growth, _debug=False):
    '''Generator function which iterates through all cohorts extant at spend_per,
    yielding spend adjusted for cohort growth and terminal change.  
    Designed to be consumed with sum(self._spender(spend_per)).
    '''
    prof_len = len(profile)
    term_value = profile[-1]
    
    # NB - USES ZERO-BASED INDEXING FOR EVERYTHING
    
    # First, define the frame by going back from the spend period by the number of launch periods
    # This frame must give the actual indexes of relevant elements of the profile for iterating
    # Zero based, and wants to go one past the end (because that's how iteration works)
    
    frame = (max(0, spend_per - launch_pers), spend_per+1)   
    
        # eg if spend_per is 5 (6 periods) and there are 2 cohorts, then want indices 4-5
        # so frame is (3,6)
    
    f_start, f_end = frame
    last_coh = min(launch_pers, spend_per) # minus 1 to make zero based
    
    if _debug:
        print("Frame: ", frame, "   Number of cohorts existing:  ", last_coh+1)
        titles = "Prof point, Coh id, Raw val, Coh adj, Term adj, Total adj,   Val".split(",")
        pads = [len(t)-1 for t in titles]
        print(" |".join(titles))

    # Iterate through the frame (i.e. through the cohorts), yielding the spend at the
    # corresponding profile point, adjusted as necessary. 
    # If the profile point is beyond the terminal, then calculate appropriately
   
    for i, prof_point in enumerate(range(f_start, f_end)):
        
        # Helps to know which cohort we're on. Last period in  frame corresponds to the 0th cohort.
        # So the first will correspond to the last launch period (minus 1 to make start at zero).  
        # Then iterate through with the index
        coh_id = last_coh - i
        coh_adj = (1+coh_growth)**coh_id # used to adjust for growth between cohorts
        term_adj = 1 # will be used to adjust for change after period - set to 1 initially
        
        if _debug: print(str(prof_point).rjust(pads[0]+1), "|", 
                        str(coh_id).rjust(pads[1]), "|", end="") 
                        
       
        # As long as the period is within the profile, yield the corresponding profile point
        if prof_point < prof_len: # length is one past the last index
            val = profile[prof_point] * coh_adj # adjust for cohort growth              

            
        # If the period is beyond the profile, use the terminal value
        else:
            # adjust the terminal value by the cohort growth, then for change after terminal period
            term_adj = (1+term_growth)**(prof_point - prof_len+1)
            val = term_value * coh_adj * term_adj

        if _debug: 
            if prof_point < prof_len:
                raw_val = profile[prof_point]
                term_adj_str = "-"
            else:
                raw_val = term_value
                term_adj_str = "{0:.3f}".format(term_adj)
                
            print(str(raw_val).rjust(pads[2]), " |", 
                      "{0:.3f}".format(coh_adj).rjust(pads[3]),  "|",
                      term_adj_str.rjust(pads[4]), "|", 
                      "{0:.3f}".format(term_adj*coh_adj).rjust(pads[5]), "|", 
                      "{0:.2f}".format(val).rjust(pads[6])) 
        yield val

###_________________________________________________________________________###

        
def get_forecast(profile, l_start, l_stop, coh_growth, term_growth, scale=1, 
                 proj_start=None, proj_stop=None, name='s_pd', 
                 output=None, _debug=False):
    ''' Arguments: 
           - a spend profile
           - launch start and stop periods
           - cohort growth (i.e. growth per period between cohorts)
           - terminal growth (i.e. change per period after end of profile)
           - scale to be applied to profile (scalar multiplier)
           - projection start and stop periods (i.e. what periods to return spend for
                                                 - default to launch start and stop)
           - name of forecast / set
        
        Returns a pd.Series of spend, unless set output='arr', when will return np.array
    '''
    
    # if no projection dates provided, default to the launch dates
    if proj_start is None: proj_start = l_start
    if proj_stop is None: proj_stop = l_stop


    # First set up some variables
    plot_range = proj_stop - proj_start
    profile = profile * scale
    launch_pers = l_stop-1 - l_start
    
    # _debugging - create and print a record
    if _debug:
        info = {"Name": name,
         "First launch period": l_start,
         "Last launch period": l_stop,
         "Profile length": len(profile),
         "Profile scaling": scale,
         "Profile max value (scaled)": max(profile),
         "Profile max period": profile.argmax(axis=0),
         "Terminal value": profile[-1],
         "Cohort growth rate": coh_growth,
         "Terminal growth rate": term_growth,
         "Projection start period": proj_start,
         "Projection end period": proj_stop,
         "Number of periods": plot_range}

        outlist = ["Input parameters:"]
        for key in info:
            temp_string = key.ljust(27) + str(info[key]).rjust(10)
            outlist.append(temp_string)
        print("\n- ".join(outlist))
    
    # Main work - make an empty np.array, fill with calls to sum(spender())
    # Wrap in pd.Series and return
    np_out = np.empty(plot_range)
    for i, per in enumerate(range(proj_start-l_start, proj_stop-l_start)):
        # if _debug: print("\nGetting period ", per)
        np_out[i] = sum(spender(per, profile, launch_pers, 
                                coh_growth, term_growth, _debug=False))
        # if _debug: print("--> Period ", per, "result: ", np_out[i])

    if output == 'arr':
        return np_out

    return pd.Series(np_out, index=range(proj_start, proj_stop), name=name)

###_________________________________________________________________________###

def get_forecast1(shape, terminal_gr_pm, cohort_gr_pm, n_pers, name=None, _debug=False):
    '''Simpler version of get_forecast().  Not yet migrated to this though.

    '''
    
    # make starting array - the shape, extended for the number of periods
    terminal_per = (1 + terminal_gr_pm) ** np.arange(1,n_pers - len(shape) +1) * shape[-1]
    if _debug: print('gf1 shape ', shape)
    if _debug: print('gf1 terminal_per ', terminal_per)
    base_arr = np.concatenate([shape, terminal_per]) 
    
    # instantiate an array to build on, adding layers (copy of base_arr)
    res = base_arr.copy()
    
    # use a factor to reflect cohort growth
    growth_factor = 1
    
    if _debug: df_out = pd.DataFrame(base_arr, columns=[0])
    
    # iterate through remaining periods (already have zero)
    for per in range(1, n_pers):
        # first calculate the new growth factor
        growth_factor *= (1 + cohort_gr_pm)
        
        # make a layer, shifting base to right and adding zeroes at start
        layer = np.concatenate([np.zeros(per), base_arr[:-per]]) * growth_factor
        if _debug: df_out[per] = layer

        res += layer
    
    if _debug:
        df_out['dfsum'] = df_out.sum(axis=1)
        df_out['result'] = res
        df_out['diff'] = df_out['dfsum']  - df_out['result'] 
        return df_out
    
    else: return pd.Series(res, name=name)


###_________________________________________________________________________###

def find_launch(sales, *, threshold, zero_first=True):
    '''Taking a list as input, plus threshold percentage of max sales,
    returns the index of the first element to exceed that threshold.
    
    If not found returns "no data"
    
    zero_first=True means index will be returned based on zero as first element.
    
    Works with tuples and pd.Series
    '''
    launch = 0
    
    max_sales = sales.max()  
    sales_threshold = threshold * max_sales
    
    try:
        for i, val in enumerate(sales):
            if i == 0 and val > sales_threshold:
                break
            elif i > 0 and val > sales_threshold:
                launch = i
                break
    except:
        return "No data"

    if launch == 0: return 0
    else:
        if zero_first: return launch
        else: return launch + 1


###_________________________________________________________________________###


