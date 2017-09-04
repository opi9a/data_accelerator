# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 14:50:32 2017

@author: GRoberta
"""

import pandas as pd
import numpy as np


def variablise(string):
    
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
                except: return string


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
            - index names:value pairs to be sliced (in any order)
            
    returns a pd.IndexSlice with the desired spec
            
    eg, if index contains the boolean 'is_cool' and ints 'year'
       'is_cool = True' will generate a slice where 'is_cool'  is set to True
       'year = slice(2006, 2009, None)' will select years 2006 to 2009 
       'year = slice(2006, None, None)' will select years 2006 onward 
       'year = [2008, 2012]' will select just those two years
       
    simply print the returned output to see what's going on
    '''
    # first turn any None entries of the input into slices
    for i in in_dict:
        print(' in get slice, with key ', i, " value", in_dict[i])
        if in_dict[i] is '' or in_dict[i] is None:
            print('setting element to noneslice')
            in_dict[i] = slice(None, None, None)
            
        else: print('it was not none')
    print('after processing the dict is ', in_dict)

    return tuple((in_dict.get(name, slice(None,None,None)))
                     for name in df.index.names)
    
 ###_________________________________________________________________________###

   
# Core spend per period function
# Given input profile, timings etc, returns the spend on each cohort.  
# It's a generator, intended to be consumed as `sum(spender(...))`.

def spender(spend_per, profile, launch_pers, coh_growth, term_growth, debug=False):
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
    
    if debug:
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
        
        if debug: print(str(prof_point).rjust(pads[0]+1), "|", 
                        str(coh_id).rjust(pads[1]), "|", end="") 
                        
       
        # As long as the period is within the profile, yield the corresponding profile point
        if prof_point < prof_len: # length is one past the last index
            val = profile[prof_point] * coh_adj # adjust for cohort growth              

            
        # If the period is beyond the profile, use the terminal value
        else:
            # adjust the terminal value by the cohort growth, then for change after terminal period
            term_adj = (1+term_growth)**(prof_point - prof_len+1)
            val = term_value * coh_adj * term_adj

        if debug: 
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


        
def get_forecast(profile, l_start, l_stop, coh_growth, term_growth, scale, 
                 proj_start, proj_stop, name='s_pd', debug=False):
    ''' Arguments: 
           - a spend profile
           - launch start and stop periods
           - cohort growth (i.e. growth per period between cohorts)
           - terminal growth (i.e. change per period after end of profile)
           - scale to be applied to profile (scalar multiplier)
           - projection start and stop periods (i.e. what periods to return spend for)
           - name of forecast / set
        
        Returns a pd.Series of spend'''
    
    # First set up some variables
    plot_range = proj_stop - proj_start
    profile = profile * scale
    launch_pers = l_stop-1 - l_start
    
    # Debugging - create and print a record
    if debug:
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
        if debug: print("\nGetting period ", per)
        np_out[i] = sum(spender(per, profile, launch_pers, 
                                coh_growth, term_growth, debug=debug))
        if debug: print("--> Period ", per, "result: ", np_out[i])

    return pd.Series(np_out, index=range(proj_start, proj_stop), name=name)

###_________________________________________________________________________###




###_________________________________________________________________________###



###_________________________________________________________________________###

