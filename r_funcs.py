# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import projection_funcs

profs_table={'prof1':'c:/Users/groberta/Work/data_accelerator/profiles/prof1.csv',
             'prof_old': "c:/Users/groberta/Work/data_accelerator/profiles/prof_old.csv"}

##_________________________________________________________________________##

'''Suite of functions with common API which can be passed to a RuleSet object

TODO: document the API etc
'''

##_________________________________________________________________________##

def trend(prod, interval, *, launch_cat=None, life_cycle_per=0,
          uptake_dur=120, plat_dur=24, gen_mult=0, term_rate_pa=0,
          threshold_rate=0.001, n_pers=12, 
          _out_type='array', _debug=False, name=None):

    '''Takes input array, with parameters, and returns a trend-based projection.

    Parameters:

        interval    The number of periods (back from last observation) that are used
                    to calculate the trend
    '''
    
    lpad, rpad = 50, 20

    if name is not None:
        if _debug: print('\n\nIn ',  name)
    else: print('\nNo name')
    
    # make sure initial array is ok
    try:
        prod = np.array(prod)
        if _debug: print("made np array, len:".ljust(lpad), str(len(prod)).rjust(rpad))
    except:
        if _debug: print("could not make np array")
        if _debug: print("prod type:".ljust(lpad), str(type(prod)).rjust(rpad))
        if _debug: print("prod len:".ljust(lpad), str(len(prod)).rjust(rpad))
            

    # make an annual moving average array
    prod[np.isnan(prod)]=0
    prod_ma = projection_funcs.mov_ave(prod, 12)
    
    # get set of info

    i_dict = {}
    i_dict['__s0'] = "spacer"

    # first classify to phase

    if life_cycle_per <= uptake_dur:
        i_dict['phase'] = 'uptake'

    elif life_cycle_per <= uptake_dur + plat_dur:
        i_dict['phase'] = 'plateau'

    else: i_dict['phase'] = 'terminal'

    # get sales info

    i_dict['__s1'] = "spacer"
    i_dict['max_sales'] = np.nanmax(prod)    
    i_dict['max_sales_per'] = np.nanargmax(prod)
    i_dict['last_spend'] = (prod[-1])

    i_dict['__s2'] = "spacer"
    i_dict['max_sales_ma'] = np.nanmax(prod_ma)    
    i_dict['max_sales_ma_per'] = np.nanargmax(prod_ma)
    i_dict['last_spend_ma'] = (prod_ma[-1])

    # get drop from max 
    i_dict['__s3'] = "spacer"
    i_dict['total_drop_ma'] = max(0, i_dict['max_sales_ma'] - i_dict['last_spend_ma'])
    i_dict['total_drop_ma_pct'] = i_dict['total_drop_ma']/i_dict['max_sales_ma']
    
    # get linear change per period over interval of periods
    # TODO calculate this on recent averages
    i_dict['__s4'] = "spacer"
    i_dict['interval'] = min(interval, len(prod))
    i_dict['interval_delta'] = prod[-1] - prod[-(1 + interval)]
    i_dict['interval_rate'] = i_dict['interval_delta'] / interval
    i_dict['interval_rate_pct'] = i_dict['interval_rate'] / prod[-(1 + interval)]
    i_dict['__endint'] = "spacer"


    # decide what this is
    # ifperiod
    
    
    # just generate projection if rate over a threshold
    
    # first sort out all the periods

    out = np.array([i_dict['last_spend_ma']]) # this period overlaps with past

    if i_dict['phase'] == 'terminal':
        # simplifies, plus avoids applying gen_mult if already in terminal
        out = out[-1] * ((1 + term_rate_pa/12) ** np.arange(1, n_pers+1))

    else:

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
        term_out = out[-1] * gen_mult * ((1 + term_rate_pa/12) ** np.arange(1, i_dict['term_pers']+1))


        out = np.append(out, term_out)


    # if i_dict['interval_rate_pct'] > threshold_rate:
    #     if _debug: print('rate is OVER threshold')
    #     out = i_dict['last_spend_ma'] + (i_dict['interval_rate'] * np.arange(1,n_pers+1))
        
    #     # apply plateau
            
    # else:
    #     if _debug: print('rate is UNDER threshold')
    #     out = i_dict['last_spend'] * np.ones(n_pers)
    

    
    if _debug: 
        i_dict['__s5'] = "spacer"
        for i in i_dict:
            if i_dict[i]=="spacer": print("")
            else:
                print(i.ljust(lpad), str(i_dict[i]).rjust(rpad))

    if _out_type == 'dict':
        i_dict['prod_ma'] = prod_ma
        i_dict['projection'] = out
        return i_dict

    elif _out_type == 'df':
        print('df output')
        spacer = np.empty(len(prod))
        spacer[:] = np.nan
        out=np.insert(out, 0, spacer)
        df=pd.DataFrame([prod, prod_ma, out], index=['raw', 'mov_ave', 'projected']).T
        return df

    else:
        return out[1:]


##_________________________________________________________________________##

def r_trend(df, n_pers, *, uptake_dur=None, plat_dur=None, gen_mult=None, term_rate_pa=None,
          threshold_rate=0.001, _interval=24, _debug=False):
    
    '''iterates through an input df
    applies trend()
    returns 
    '''
    out=[]

    for row in df.itertuples():
        launch_date = row[0][df.index.names.index('start_month')] # gets the index number of 'start_date' in the df.index list
        launch_date = pd.Period(launch_date, freq='M')
        last_date = df.columns[-1]
        life_cycle_per = last_date - launch_date
        print(launch_date, last_date, life_cycle_per)

        out_array = trend(row[1:], _interval, n_pers=n_pers, life_cycle_per=life_cycle_per,
                        uptake_dur=uptake_dur, plat_dur=plat_dur, gen_mult=gen_mult, 
                        name=row[0][0], term_rate_pa=term_rate_pa,  
                        _out_type='array', _debug=_debug)

        print(out_array[-10:])
        out.append(out_array)
        # call trend on row[1:]
        # append to out

 
    # if _debug:
    #     pad1 = 20
    #     print("launch_date".ljust(pad1), launch_date, 
    #           "\nlast_date".ljust(pad1), last_date, 
    #           "\nn_pers".ljust(pad1), n_pers, 
    #           "\nlast_spend".ljust(pad1), last_spend, 
    #           "\nstart_x".ljust(pad1), start_x, 
    #           "\nscaling_f".ljust(pad1), scaling_f)

    # Build the df index and columns

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

def r_terminal(df, n_pers, *, term_rate_pa, _debug=False):
    '''take average of last 3 monthly values and extrapolate at terminal rate
    
    default args:   data_in, n_pers
    
    r_args:         term_rate_pa     - terminal rate, annual (positive)
    
    return:         dataframe
    
    '''
    # make an initial np array with growth, based on index-1 = 1

    term_rate_mo = float(term_rate_pa) / 12
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
          coh_growth_pa, term_growth_pa, name='future', _debug=True):
    
    coh_growth = coh_growth_pa /12
    term_growth = term_growth_pa /12

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
    fut = projection_funcs.get_forecast(profile, l_start, l_stop, coh_growth,term_growth,1, proj_start-1, proj_stop, name=name)
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