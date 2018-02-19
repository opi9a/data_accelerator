# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import projection_funcs as pf
import policy_tools as pt
import inspect
from pprint import pprint

profs_table={'prof1':'c:/Users/groberta/Work/data_accelerator/profiles/prof1.csv',
             'prof_old': "c:/Users/groberta/Work/data_accelerator/profiles/prof_old.csv"}

##_________________________________________________________________________##

'''Suite of functions with common API which can be passed to a RuleSet object.

Actual functions for passing all prefixed `r_`.  Others (top of file) are auxiliary.

TODO: fully document the API.  Basically need to be able to pass in a df with rows 
for spend on products, and return a df with forecasts for each product (using original
index, so can be appended / joined to input).

Will be a bunch of other parameters, but these are automatically detected in the RuleSet object,
and passed appropriately to front end.

'''
##_________________________________________________________________________##

def infer_launch(in_arr, max_spend, streak_len_threshold=12, delta_threshold = 0.2, 
                    _ma_interval=12, verbose_return=False, _debug=False):

    '''Infers a launch date, given a trajectory of spend, by identifying an uptake phase - 
        the most recent streak of consistent growth - and returning a dict describing it.


    PARAMETERS:

        in_arr                  input array (numpy I think)
        max_spend               the individual product's max spend value
        streak_len_threshold    the number of periods of successive growth
        delta_threshold         the change (proportionate to max spend) required for a streak to qualify
        _ma_interval            interval for calculatino of moving averages
        verbose_return          [bool] return an object with info on all streaks, rather than the one detected


    RETURN: 
        dictionary with:
            uptake_detected:    [bool] if a qualifying streak was found
            start:              start of the detected qualifying streak
            delta:              the increase over the uptake streak (proportion of max spend)
            inferred_launch:    expressed relative to period zero of in_arr


    '''

    
    # first get the moving ave. and ma_diffs
    ma = pf.mov_ave(in_arr, _ma_interval)
    ma_diff = ma - np.insert(ma, 0, np.zeros(_ma_interval))[:-_ma_interval]
          
    # now look for streaks
    streak = 0
    streaks = {'uptake_detected':False, 'streaks':{}}
    for p in range(_ma_interval, len(ma_diff)):   
        if _debug: print(p, end="")
        if ma_diff[p] > 0 and p < len(ma_diff)-1:
            streak += 1
            if _debug: print("+", end="")
        else:
            if _debug: print('-', end="")
            if streak>0 and _debug:
                print('\nstreak ends, length is: ', streak, end=" ")
            if streak > streak_len_threshold:
                if _debug: print(', OVER length threshold, ', end=" ")

                # calculate the delta, based on the mov ave
                if _debug: print(', START: ', p-streak, ma[p-streak], end=" ")
                if _debug: print(', END: ', p, ma[p], end=" ")
                delta = (ma[p] - ma[p-streak]) / max_spend
                if _debug: print(', delta is, ', delta, end=" ")
                
                # only add to dict if is over threshold
                if delta > delta_threshold:
                    if _debug: print('streak is OVER delta threshold, delta is:'.ljust(20), delta)
                    s_key = "s_"+str(p)
                    streaks['streaks'][s_key]={}
                    streaks['streaks'][s_key]['end_per'] = p
                    streaks['streaks'][s_key]['length'] = streak
                    streaks['streaks'][s_key]['end_val'] = ma[p]
                    streaks['streaks'][s_key]['start_val'] = ma[p-streak]          
                    streaks['streaks'][s_key]['raw_delta'] = ma[p] - ma[p-streak]
                    streaks['streaks'][s_key]['prop_delta'] = delta
                               
                    # only going to return the last qualifying streak (others available for `verbose_return=True`)
                    streaks['uptake_detected'] = True
                    streaks['last_per'] = p
                    streaks['last_per_len'] = streak
                    streaks['last_delta'] = delta 
                
                else:
                    if _debug: print("delta of ", delta, " is BELOW threshold of ", delta_threshold, "\n")
                    
            # terminate the streak even if it doesn't qualify
            streak = 0
    if _debug: pprint(streaks)        
    # infer start, using y=mx equation of straight line to get x1 - the offset of the first point from origin
    if streaks['uptake_detected']:
        streak_start = streaks['last_per'] - streaks['last_per_len']
        window = min(streaks['last_per_len'], _ma_interval)
        # set lower point at start of streak
        y1 = ma[streak_start]
        if _debug: print("y1 - low val - is", y1)
        y2 =  ma[streak_start + window]
        if _debug: print("y2 - high val - is", y2)
        dy = y2 - y1
        dx = window
        x1 = (dx/dy)*y1
    
        # move the offset back to account for ma window 
        inferred_launch = streak_start -int(x1) - (_ma_interval//2)
        if _debug: print('inferred_launch is ', inferred_launch)
        
    if verbose_return:
        return(streaks)
    
    elif streaks['uptake_detected']:
        return dict(uptake_detected = True,
                    start=streaks['last_per'] - streaks['last_per_len'],
                    end=streaks['last_per'],
                    delta=streaks['last_delta'],
                   inferred_launch=inferred_launch)
    else:
        return dict(uptake_detected = False)

##_________________________________________________________________________##

def trend(prod, interval, *, launch_cat=None, life_cycle_per=0,
          shed=None, uptake_dur=120, plat_dur=24, gen_mult=0, term_gr_pa=None,
          term_gr = 0,  threshold_rate=0.001, n_pers=12, 
          _out_type='array', _debug=False, name=None):

    '''Takes input array, with parameters, and returns a trend-based projection.

    Parameters:

        prod        An input array of spend

        interval    The number of periods (back from last observation) that are used
                    to calculate the trend
    '''

    # _debug = True

    if _debug: print("\nIN FUNCTION:  ".ljust(20), inspect.stack()[0][3])
    if _debug: print("..called by:  ".ljust(20), inspect.stack()[1][3], end="\n\n")

    pad, lpad, rpad = 30, 30, 20

    if _debug: 
        print('\nInput name(?) '.ljust(lpad),  name)
        print('type'.ljust(lpad), type(prod))

    # make sure initial array is ok
    if isinstance(prod, pd.Series):
        prod = np.array(prod)
        if _debug: print('found a series, len'.ljust(pad), len(prod))

    elif isinstance(prod, tuple):
        prod = np.array(prod)
        if _debug: print('found a tuple, len'.ljust(pad), len(prod))

    elif isinstance(prod, np.ndarray):
        prod = np.array(prod)
        if _debug: print('found an ndarray, len'.ljust(pad), len(prod))
        if len(prod) ==1:
            if _debug: print('unpacking array of len 1')
            prod = prod[0]
            if _debug: print('array len now'.ljust(pad), len(prod))

    else:
        print("DON'T KNOW WHAT HAS BEEN PASSED - make sure its not a dataframe")
        return

    if _debug: 
        print('\nhead of prod')
        print(prod[:12], '\n')

    if shed is not None:
        if _debug: print('using passed shed:')
        uptake_dur = shed.uptake_dur
        plat_dur = shed.plat_dur
        gen_mult = shed.gen_mult

    if _debug: 
        print(" - uptake_dur".ljust(lpad), uptake_dur)
        print(" - plat_dur".ljust(lpad), plat_dur)
        print(" - gen_mult".ljust(lpad), gen_mult)
        print("lifecycle period".ljust(lpad), life_cycle_per)

    if term_gr_pa is not None:
        term_gr = term_gr_pa/12

    # make an annual moving average array
    prod[np.isnan(prod)]=0
    prod_ma = pf.mov_ave(prod, 12)
    
    # get set of info

    i_dict = {}
    i_dict['__s0'] = "spacer" # just for printing the dict more nicely

    # first classify to phase

    if life_cycle_per <= uptake_dur:
        i_dict['phase'] = 'uptake'

    elif life_cycle_per <= uptake_dur + plat_dur:
        i_dict['phase'] = 'plateau'

    else: i_dict['phase'] = 'terminal'

    # get spend info

    i_dict['__s1'] = "spacer"
    i_dict['max_spend'] = np.nanmax(prod)    
    i_dict['max_spend_per'] = np.nanargmax(prod)
    i_dict['last_spend'] = (prod[-1])

    i_dict['__s2'] = "spacer"
    i_dict['max_spend_ma'] = np.nanmax(prod_ma)    
    i_dict['max_spend_ma_per'] = np.nanargmax(prod_ma)
    i_dict['last_spend_ma'] = (prod_ma[-1])

    # get drop from max 
    i_dict['__s3'] = "spacer"
    i_dict['total_drop_ma'] = max(0, i_dict['max_spend_ma'] - i_dict['last_spend_ma'])
    if not i_dict['max_spend_ma'] == 0:
        i_dict['total_drop_ma_pct'] = i_dict['total_drop_ma']/i_dict['max_spend_ma']
    
    # get linear change per period over interval of periods
    # TODO calculate this on recent averages
    i_dict['__s4'] = "spacer"
    i_dict['interval'] = min(interval, len(prod))
    i_dict['interval_delta'] = prod[-1] - prod[-(1 + interval)]
    i_dict['interval_rate'] = i_dict['interval_delta'] / interval
    if not prod[-(1 + interval)] == 0:
        i_dict['interval_rate_pct'] = i_dict['interval_rate'] / prod[-(1 + interval)]
    i_dict['__endint'] = "spacer"



    # first sort out all the periods

    out = np.array([i_dict['last_spend_ma']]) # this period overlaps with past, will be snipped later

    if i_dict['phase'] == 'terminal':
        # simplifies, plus avoids applying gen_mult if already in terminal
        # different to 
        if _debug: print('adding terminal phase')
        out = out[-1] * ((1 + term_gr) ** np.arange(1, n_pers+1))
        if _debug: print('out - terminal'.ljust(pad), out)

    else:

        if _debug: print('finding pre-terminal phases')

        i_dict['uptake_pers'] = min(max(uptake_dur - life_cycle_per, 0), 
                                                    n_pers - (len(out)-1))
        uptake_out = out[-1] + (i_dict['interval_rate'] 
                                                * np.arange(1,i_dict['uptake_pers']))
        life_cycle_per += i_dict['uptake_pers']

        out = np.append(out, uptake_out)
        i_dict['plat_pers'] = min(max((uptake_dur + plat_dur) - life_cycle_per, 0),
                                                    n_pers - (len(out)-1))
        plat_out = out[-1] * np.ones(i_dict['plat_pers'])
        life_cycle_per += i_dict['plat_pers']

        out = np.append(out, plat_out)

        i_dict['term_pers'] = max(n_pers - (len(out)-1), 0)
        term_out = out[-1] * gen_mult * ((1 + term_gr) ** np.arange(1, i_dict['term_pers']+1))

        out = np.append(out, term_out)

        # eliminate any negatives
        out[out<0] = 0


   
    if _debug: 
        i_dict['__s5'] = "spacer"
        for i in i_dict:
            if i_dict[i]=="spacer": print("")
            else:
                print(i.ljust(lpad), str(i_dict[i]).rjust(rpad))

    if _out_type == 'dict':
        i_dict['prod_ma'] = prod_ma
        i_dict['projection'] = out
        if _debug: print("\LEAVING:  ", inspect.stack()[0][3])
        return i_dict

    elif _out_type == 'df':
        if _debug: print('df output')
        spacer = np.empty(len(prod))
        spacer[:] = np.nan
        out=np.insert(out, 0, spacer)
        df=pd.DataFrame([prod, prod_ma, out], index=['raw', 'mov_ave', 'projected']).T
        if _debug: print("\LEAVING:  ", inspect.stack()[0][3])
        return df

    else:
        if _debug: print("\nLEAVING:  ", inspect.stack()[0][3])
        return out[1:]


##_________________________________________________________________________##

def r_trend(df, n_pers, *, shed=None, uptake_dur=None, plat_dur=None, gen_mult=None, term_gr=0, 
            loe_delay=None, use_eff_loe=True, 
            threshold_rate=0.001, _interval=24, _out_type='array', _debug=False):
    
    '''Iterates through an input df, applying trend(), returning a df of projections.

    Key logic is calculation of lifecycle period, which is passed to trend() to orient the projection.
    This is currently done with reference to the loe date.  

    Eg if last observation is 1-2017, and loe date for a product is 1-2020, then the lifecycle period is 
    36 periods before the loe lifecycle period (which is uptake_dur + plat_dur).

    So if uptake_dur=56, plat_dur=100, lifecycle period is 120 (56+120-36).  When passing to trend(), another
    36 periods of plateau will be projected, and then the loe drop will be applied.

    To reflect a lag in erosion, therefore need to position product further back in lifecyle.  
    In above example, if lag was 6m, pass the lifecycle period of 114, so that 42 periods of plateau are applied.

    To do this, can set a variable in the df for eff_loe_month (with conditional flag use_eff_loe).  
    Also have facility to accept an loe_delay parameter.  
    Could include this loe_delay parameter in the lifecycle model.

    _out_type='array' specifies that trend() returns an array, obv, which is required for the actual projections.
    But can pass 'df' to get the dataframe output (showing mov ave etc) if calling to visualise projections etc. 
 
    '''
    if _debug: print("\nIN FUNCTION:  ".ljust(20), inspect.stack()[0][3])
    if _debug: print("..called by:  ".ljust(20), inspect.stack()[1][3], end="\n\n")

    pad = 25
    out=[]

    #  housekeeping - assign lifecycle variables depending on what was passed
    if shed is not None:
        if _debug: print('using passed shed:')
        uptake_dur = shed.uptake_dur
        plat_dur = shed.plat_dur
        gen_mult = shed.gen_mult

    # define the lifecycle period in which loe will occur, according to input shed or shape data
    # - this is used to fix the actual lcycle period
    loe_lcycle_per = uptake_dur + plat_dur

    # enter the key loop through rows in the input df
    for row in df.itertuples():

        # TODO make params a dict, rather than have to look up by index number
        params = row[0]
        data = row[1:]

        if _debug: print('\nMolecule'.ljust(pad), params[df.index.names.index('molecule')])
        if _debug: print('Setting'.ljust(pad), params[df.index.names.index('setting')])

        # get loe month - look for eff_loe_month first
        if _debug: print('look for eff loe?'.ljust(pad), use_eff_loe)
        if _debug: print('eff_loe_month in params?'.ljust(pad), 'eff_loe_month' in df.index.names)
        if use_eff_loe and 'eff_loe_month' in df.index.names:
            loe_month = params[df.index.names.index('eff_loe_month')]; 
            if _debug: print('found eff loe month'.ljust(pad), loe_month)


        # if not, just take the loe_date and turn into a month
        else: 
            if _debug: print('taking raw loe date'.ljust(pad), end=" ")
            loe_month = pd.Period(params[df.index.names.index('loe_date')], freq='M'); 

        # now look for an loe_delay parameter
        if loe_delay is not None:
            loe_month += loe_delay
            if _debug: print('adding loe_delay of '.ljust(pad), loe_delay)
        
        if _debug: print(loe_month)

        # get start month - PRETTY MUCH DEPRECATED AS GO OFF LOE NOW
        start_month = params[df.index.names.index('start_month')] # gets the index number of 'start_date' in the df.index list
        start_month = pd.Period(start_month, freq='M')
        if _debug: print('start_month'.ljust(pad), start_month)

        # get date of last month in actual data
        last_month = df.columns[-1]
        if _debug: print('last_month'.ljust(pad), last_month)

        # get time after / before loe (negative if befote loe)
        pers_post_loe = last_month - loe_month
        if _debug: print('pers_post_loe'.ljust(pad), pers_post_loe)

        # infer the lifecycle period from this
        life_cycle_per = loe_lcycle_per + pers_post_loe
        if _debug: print('life_cycle_per'.ljust(pad), life_cycle_per)

        # call trend
        out_array = trend(data, _interval, n_pers=n_pers, life_cycle_per=life_cycle_per, shed=shed,
                        name=params[0], term_gr=term_gr,  
                        _out_type=_out_type, _debug=False)

        out.append(out_array)

    # just return this out list of dataframes if passing 'df' as _out_type (eg for visualisations)
    if _out_type == 'df':
        return out

    # Otherwise build the df index and columns and return a single df

    cols = pd.PeriodIndex(start=last_month+1, periods=n_pers, freq='M')

    if _debug: print("\LEAVING:  ", inspect.stack()[0][3])

    return pd.DataFrame(out, index=df.index, columns=cols)



##_________________________________________________________________________##

def r_trend_old(df, n_pers, *, streak_len_thresh=12, delta_thresh = 0.2,
        uptake_dur=90, plat_dur=24, gen_mult=0.9, term_gr_pa=0,
          threshold_rate=0.001, ma_interval=12, _debug=False):
    
    '''iterates through an input df
    applies trend()
    returns 
    '''
    out=[]

    for row in df.itertuples():

        max_spend = row[0][df.index.names.index('max_spend')]
        inferred_launch_output = infer_launch(row[1:], max_spend=max_spend, delta_threshold = delta_thresh,
                                                 _ma_interval=ma_interval, _debug=_debug)

        # print(inferred_launch_output)

        last_date = df.columns[-1]

        if inferred_launch_output['uptake_detected']: 
            
            launch_date = pd.Period(df.columns[0], freq='M') +  inferred_launch_output['inferred_launch']
            life_cycle_per = last_date - launch_date

            if _debug: print("for ", row[0][0], " inferred launch is ", launch_date, ", life_cycle_per is ", life_cycle_per)

            out_array = trend(row[1:], ma_interval, n_pers=n_pers, life_cycle_per=life_cycle_per,
                            uptake_dur=uptake_dur, plat_dur=plat_dur, gen_mult=gen_mult, 
                            name=row[0][0], term_gr_pa=term_gr_pa,  
                            _out_type='array', _debug=_debug)


        else:
            print("for ", row[0][0], " NO inferred launch")           
            out_array = trend(row[1:], ma_interval, n_pers=n_pers, life_cycle_per=0,
                uptake_dur=0, plat_dur=0, gen_mult=gen_mult, 
                name=row[0][0], term_gr_pa=term_gr_pa,  
                _out_type='array', _debug=_debug)

        if _debug: print(out_array[-10:])
        out.append(out_array)
        # call trend on row[1:]
        # append to out


    cols = pd.PeriodIndex(start=last_date+1, periods=n_pers, freq='M')
    
    return pd.DataFrame(out, index=df.index, columns=cols)



##_________________________________________________________________________##

def r_profile(df, n_pers, 
              *, profile, gen_mult, _debug=True):
    '''Applies a profile, with a variable multiplier for patent expiry
    
    args:   data (df or series - future) - which includes index data eg launch year
            n_pers
            r_args - to include the profile itself and the gen_mult
            
    return: df (or series - future)
    '''
    # first, if the passed profile is a string, retrieve the array
    if isinstance(profile, str):
        print("it's a string, so retrieving array")
        profile = np.genfromtxt(profs_table[profile])
        print(profile)

    out=[]

    for row in df.itertuples():
        #note the structure of each row:  [0] is the index (a tuple), [1:] is the data
        launch_date = row[0][df.index.names.index('start_month')] # gets the index number of 'start_date' in the df.index list
        launch_date = pd.Period(launch_date, freq='M')
        last_date = df.columns[-1]
        last_spend = row[-1]
        print('last, launch', last_date, launch_date)
        start_x = last_date - launch_date
        basic = profile[start_x: start_x+n_pers]
        scaling_f = last_spend / profile[start_x-1]
        y = basic * scaling_f
        out.append(y)
    
    if _debug:
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



def r_tprofile(df, n_pers, *, profile):
    '''
    Simple extrapolation using a fixed profile.
    No use of cohorts etc.
    Takes average of last 3 periods as basis
    Projects future periods according to the profile (multiplication of the basis)
    If projecting past the end of profile, just continue at level of last period
    '''
    ave_last = np.array(df.iloc[:,-6:].sum(axis=1)/6)
    print('ave last ', ave_last.shape)
    
    if isinstance(profile, str):
        print("it's a string, so retrieving array")
        profile = np.genfromtxt(profs_table[profile])

    print('profile input ', profile.shape)


    if len(profile) < n_pers:
        profile1 = np.append(profile,[profile[-1]]*(n_pers-len(profile)))
    else:
        profile1 = profile[:n_pers]
 
    print('new profile shape ', profile1.shape)
        
    profile_arr = np.array([profile1] * len(df)).T
    print('profile array ', profile_arr.shape)
    out = np.multiply(profile_arr, ave_last)
        
    ind = df.index
    last_date = df.columns[-1]
    cols = pd.PeriodIndex(start=last_date+1, periods=n_pers, freq='M')

    return pd.DataFrame(out.T, index=ind, columns=cols)



##_________________________________________________________________________##

def r_terminal(df, n_pers, *, term_gr_pa, _debug=False):
    '''take average of last 3 monthly values and extrapolate at terminal rate
    
    default args:   data_in, n_pers
    
    r_args:         term_gr_pa     - terminal rate, annual (positive)
    
    return:         dataframe
    
    '''
    # make an initial np array with growth, based on index-1 = 1

    term_gr_mo = float(term_gr_pa) / 12
    x = np.array([(1+term_gr_mo)**np.arange(1,n_pers+1)]*len(df))
    ave_last = df.iloc[:,-3:].sum(axis=1)/3
    x=x*ave_last[:,None]
    
    # Build the df index and columns
    ind = df.index
    last_date = df.columns[-1]
    cols = pd.PeriodIndex(start=last_date+1, periods=n_pers, freq='M')
    
    return pd.DataFrame(x, index=ind, columns=cols)


##_________________________________________________________________________##

def r_fut(df, n_pers, *, profile, cutoff_date,
          coh_gr_pa, term_gr_pa, name='future', _debug=True):
    
    coh_gr = coh_gr_pa /12
    term_gr = term_gr_pa /12

    # check if profile is an array yet
    if isinstance(profile, str):
        print("it's a string, so retrieving array")
        profile = np.genfromtxt(profs_table[profile])
        print(profile)


    # work out what l_start and l_stop should be, and how passed to get_forecast()

    # sum the df, as we don't care about individual products.  NB it's now a Series
    
    cutoff_date = pd.Period(cutoff_date)
    df=df.sum()
    last_date = df.index[-1]

    l_start=0
    l_stop=(last_date - cutoff_date)+n_pers
    proj_start=(last_date - cutoff_date)
    proj_stop=(last_date - cutoff_date)+n_pers
    
    if _debug: 
        pad=20
        print("n_pers".ljust(pad), n_pers,"\nlast_date".ljust(pad), last_date,"\ncutoff_date".ljust(pad), cutoff_date,"\nl_start".ljust(pad), l_start,"\nl_stop".ljust(pad), l_stop,"\nproj_start".ljust(pad), proj_start,"\nproj_stop".ljust(pad), proj_stop)
    
    # note this gets a projection for one period behind the required, to allow scaling.  This must be sliced off later.
    fut = pf.get_forecast(profile, l_start, l_stop, coh_gr,term_gr,1, proj_start-1, proj_stop, name=name)
    if _debug: print(fut)
        
    # now get scaling factor. Take cumulative sum for last period in slice passed
    # (also should have available the actual period from the slice - do later)
    last_sum = df[-1]

    # to scale, want the period just before the fut forecast to equal last_sum.  Deliberately overlap, and then snip off first period?
    scaler=last_sum/fut.iloc[0]
    
    if _debug: print(last_sum, scaler)
    
    out = (fut*scaler)[1:]
    
    out.index=pd.PeriodIndex(start=last_date+1, periods=n_pers, freq='M')

    return pd.DataFrame(out)



##_________________________________________________________________________##



def r_fut_tr(df, n_pers, *, cut_off, shed=None, loe_delay=None,
             coh_gr_pa=None, coh_gr=None, term_gr_pa=None, term_gr=None, name='future', _debug=False):
    
    '''Generates a projection of spend on future launches, based on cumulation
    of a lifecycle profile (itself imputed from observations), and scaled using observations.

    Note only returns the future projection, not the input past observations

    1. Make a shape corresponding to the passed shed
    2. Use this to project a forecast (unscaled)
    3. Scale the forecast to the last period of actual observations
    4. Slice the forecast to give the future only

    '''

    pad = 25

    # INCREMENT PLAT_DUR BY LOE DELAY before passing to make_shape1()
    # The non future trend functions use this differently, as they need to calculate the loe_month for
    # extending observations etc, based on observed loe date.  Think it's ok this way but need to be aware.

    # for future want to probably include this as part of shed.  Though there's an argument it's
    # really part of the plat_dur (in effect), and the need to fiddle around is only when you are working out 
    # the plat_dur from an external loe date.
    
    if loe_delay is not None:
        if _debug: print('remaking shed to add loe delay\n')
        shed = pt.Shed(shed.shed_name + '_1', 
                       shed.uptake_dur,
                       shed.plat_dur + loe_delay,
                       shed.gen_mult)
        if _debug: 
            print("shed now")
            print(shed)
   


    if term_gr_pa is not None:
        term_gr = term_gr_pa / 12

    if coh_gr_pa is not None:
        coh_gr = coh_gr_pa / 12

    # will be working with the sum of the input df
    df=df.sum()

    # get useful dates
    cut_off = pd.Period(cut_off)
    last_date = df.index[-1]


    # 1. Make a shape from the passed shed
    shape = pt.make_shape1(shed=shed)

    # 2. Use this to project a forecast.
    # - need to project for n_pers plus the overlap with actual
    overlap = last_date - cut_off
    if _debug: 
        print('cut off:'.ljust(pad), cut_off)
        print('last_date:'.ljust(pad), last_date)
        print('overlapping periods:'.ljust(pad), overlap)

    fut = pf.get_forecast1(shape, term_gr=term_gr, coh_gr=coh_gr, n_pers=n_pers+overlap, name=name)

    # 3. Scale the forecast
    #  Take cumulative sum for last period in slice passed
    last_sum = df[-1]
    if _debug: print('spend at last period'.ljust(pad), last_sum)

    # to scale, want the period just before the fut forecast to equal last_sum.  
    if _debug: print('spend at overlap period'.ljust(pad), fut.iloc[overlap])
    scaler=last_sum/fut.iloc[overlap-1]
    if _debug: print('scaler to apply'.ljust(pad), scaler)
    
    fut = (fut*scaler)
    if _debug: print("\ntail of actual:\n", df.tail(), "\n")
    if _debug: print("\nscaled fut at overlap:\n", fut[overlap-5:overlap+5].head(), "\n")

    # 4. Slice the forecast to give the future only
    out = fut[overlap:]

    out.index=pd.PeriodIndex(start=last_date+1, periods=len(out), freq='M')

    return pd.DataFrame(out)