import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
import matplotlib.ticker as ticker
import projection_funcs as pf
from collections import namedtuple
import inspect


def spend_mult(sav_rate, k1=None, k2=None, ratio=None):
    '''Returns the multiplier to apply to baseline spend 
    to reflect a change in threshold (k1->k2), given the savings (drugs substitution)
    as a proportion r of the original costs

    k1 = (C1 - s)/dQ : k is cost/QALY, c is cost of drug, s is savings
    s = rC1, r=s/C1  : r is savings as proportion of initial costs
    k1 = (1-r)C1     : assume dQ is 1 as effects are all relative
    C1 = k1/(1-r)
    k2 = C2 -rC1; C2 = k2 + rC1 = k2 + rk1/(1-r)
    C2/C1 = (k2 + rk1/(1-r))*(k1/(1-r)) = (1-r)(k2/k1) + r
    '''
    
    r = sav_rate

    if ratio is None:
        if (k1 is None) or (k2 is None): 
            print('need a ratio or k1 and k2')
            return
        
        else: ratio = k2/k1

    return ((1-r)*(ratio)) + r


##_________________________________________________________________________##



def dump_to_xls(res_df, outfile, in_dict=None, shapes=None, log=False, _debug=False):
    '''Dump a results DataFrame to an xls, with option to make shapes from a  
     dict of scenarios or from a scenario alone, and/or passing shapes
    '''
    if _debug: print("\nIN FUNCTION:  ".ljust(20), inspect.stack()[0][3])
    if _debug: print("..called by:  ".ljust(20), inspect.stack()[1][3], end="\n\n")

    # initialise vars
    params_header = None
    shapes_body = None

    # first get the params header and shapes, depending on input
    # if a policy is passed
    params_header = make_params_table(in_dict, log=log, _debug=_debug)
    shapes_body = make_shapes1(in_dict, flat=True, multi_index=True)

    if _debug: print('\ngot header and shapes from in_dict:\n')
    if _debug: print("params_header\n", params_header, end="\n\n")
    if _debug: print("shapes_body\n", shapes_body.head(), end="\n")


    # if shapes are passed
    if shapes is not None:
        if _debug: print('preparing to overwrite shapes')
        shapes_body = shapes
        if _debug: print("new shapes_body\n", shapes_body.head(), end="\n")


    # assemble outputs
    shapes_out = params_header.append(shapes_body)
    main_out = params_header.append(res_df)
    annual_out = params_header.append(res_df.groupby(res_df.index.year).sum())

    # write out
    writer = pd.ExcelWriter(outfile)
    shapes_out.to_excel(writer, 'shapes')
    main_out.to_excel(writer, 'main')
    annual_out.to_excel(writer, 'annual')

    writer.save()

    if _debug: print("\LEAVING:  ", inspect.stack()[0][3])


##_________________________________________________________________________##

def dump_shapes(scens):
    '''for dictionary of scenarios, dumps a single df with all the shapes 
     - constructed from the Sheds etc, by make_shapes()

     TODO make this cope with dicts or list, with whatever depth
    '''
    
    all_shapes = pd.DataFrame()
    for s in scens:
        for l in scens[s]:
            all_shapes[s,l] = make_shapes([scens[s][l]])
    all_shapes.columns = pd.MultiIndex.from_tuples(all_shapes.columns)
    all_shapes.columns.names = ['scenario', 'spendline']
    all_shapes.index.name = 'period'
    return all_shapes


##_________________________________________________________________________##


def first_elem(in_struct):
    '''Returns the type first element of the input, be it a list or a dict
    '''

    if isinstance(in_struct, list): 
        return in_struct[0]
    
    elif isinstance(in_struct, dict):
        return in_struct[list(in_struct.keys())[0]]


##_________________________________________________________________________##


def flatten(struct, _debug=False):
    '''Return a flat dict of spendlines, keys are tuples of the hierarchical name
    '''
    out_dict = {}
    
    def _dig(in_struct, name=None):
        # if _debug: print('entering dig with', in_struct)
        if name is None: name = []
        


        if isinstance(first_elem(in_struct), dict):
            if _debug: print('in a dict')
            for s in in_struct:
                if _debug: print('digging to ', s)
                _dig(in_struct[s], name + [s])
                
        elif isinstance(first_elem(in_struct), list):
            if _debug: print('in a list')
            for s in in_struct:
                _dig(s, name + ['list'])     
                
        else: # can't get it to use isinstance to id spendline here so have to do by default :/
            if _debug: print('in a spendline I ASSUME - type ', type(first_elem(in_struct)))
            for l in in_struct:
                if _debug: print('element is ', l)
                out_dict[tuple(name + [l])] = in_struct[l]

   
    _dig(struct)

    return out_dict

###_________________________________________________________________________###


def impute_peak_spend(df_slice, cutoff, shed, coh_gr, term_gr, ave_interval=3, 
                        _debug=False):
    '''For an input df slice (NB not by launch date), and lifecycle shape, 
    returns the impled peak spend associated with the lifecycle shape
    (i.e. for an individual product or cohort).
    
    NOTE CHANGED TO MONTTLY GROWTH INPUTS

    Returns peak spend per period for one period's worth of launches.  
    So if period is month, multiply *12*12 for peak spend pa of a year of launches.
    
    ave_interval is the interval over which terminal spend in the observed set is calculated

    Works by comparing a synthetic cumulation based on the lifecycle shape with actual spend, 
    initial ('stub') spend since a cutoff date.

    First make the synthetic cumulation, scaled to peak spend = 1.
    Then align the initial period with the start date of the stub.
    Then get the actual spend at the end of the observed stub 'end_spend'.
    Compare this to the spend at the corresponding period of the synthetic cumulation
    The ratio of these is the implied lifecycle peak spend.
    '''
        
    # make sure cutoff is a period (will take a string here)
    cutoff = pd.Period(cutoff, freq='M')
    
    # work out how many periods since cutoff
    n_pers = df_slice.columns[-1] - cutoff
    if _debug: print('npers', n_pers)
    
    # get last real observations - only need spend in last period really
    ixs = pf.get_ix_slice(df_slice, dict(start_month=slice(cutoff,None,None)))
    summed = df_slice.loc[ixs,:].sum()
    
    if _debug: print('summed df tail\n', summed.tail())

    end_spend = summed[-ave_interval:].mean()
    if _debug: print('end_spend {:0,.0f}'.format(end_spend))

    # make unscaled simulated cumulation
    shape = make_profile_shape(1, shed)
    if _debug: print("shape\n", shape)

    sim = pf.get_forecast(shape, l_start=0, l_stop=n_pers, 
                          coh_growth=coh_gr, term_growth=term_gr)

    if _debug: print('tail of cumulation\n', sim.tail())

    end_sim = sim[-ave_interval:].mean()

    if _debug: print('end_sim tail', end_sim)

    return end_spend / end_sim

##_________________________________________________________________________##

def make_log(in_dict, _debug=False):
    '''Return a df with log contents of an input dict containing spendlines
    '''
    pad = 25
    if _debug: print("\nIN FUNCTION:  ".ljust(20), inspect.stack()[0][3])
    if _debug: print("..called by:  ".ljust(20), inspect.stack()[1][3], end="\n\n")

    # first get a flat dict
    flat = flatten(in_dict)

    # instantiate the dataframe with the keys
    df = pd.DataFrame(columns=flat.keys())

    # now go through each spend line, and then each log entry
    for line in flat:
        if _debug: print('now in spend line'.ljust(pad), line)
        if _debug: print('log is'.ljust(pad), flat[line].log)
        for entry in flat[line].log:
            if _debug: print('now in log entry'.ljust(pad), entry)
            df.loc[entry, line] = flat[line].log[entry]
    
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    if _debug: print('leaving ', inspect.stack()[0][3])

    return df

##_________________________________________________________________________##

def make_params_table(pol_dict, index=None, log=False, _debug=False):
    '''Constructs a dataframe with parameters for spendlines in an input dict.

    There's a default set of row names - pass your own index if reqd

    TODO auto pick up index
    '''
    if _debug: print("\nIN FUNCTION:  ".ljust(20), inspect.stack()[0][3])
    if _debug: print("..called by:  ".ljust(20), inspect.stack()[1][3], end="\n\n")

    if index is None:
        index = """peak_spend_pa peak_spend icer sav_rate 
                    uptake_dur plat_dur gen_mult launch_delay launch_stop
                    term_gr_pa term_gr_pm coh_gr_pa coh_gr_pm""".split()

    df = pd.DataFrame(index=index)

    flat_dict = flatten(pol_dict)

    for q in flat_dict:
        params = [flat_dict[q].peak_spend*12*12, # double annualised
                 flat_dict[q].peak_spend,
                 flat_dict[q].icer,
                 flat_dict[q].sav_rate,

                 int(flat_dict[q].shed.uptake_dur),
                 int(flat_dict[q].shed.plat_dur),
                 flat_dict[q].shed.gen_mult,
                 flat_dict[q].launch_delay,
                 flat_dict[q].launch_stop,

                 flat_dict[q].term_gr*12,
                 flat_dict[q].term_gr,
                 flat_dict[q].coh_gr*12,
                 flat_dict[q].coh_gr]

        df[q] = params

    
    if log: df = df.append(make_log(pol_dict, _debug=_debug))

    df.columns = pd.MultiIndex.from_tuples(df.columns)

    if _debug: print('leaving ', inspect.stack()[0][3])

    return df

#_________________________________________________________________________##

def make_profile_pts(in_points, _debug=False):
    '''For a set of input x,y points, returns a profile generated by 
    linear extrapolation.
    
    Points are sorted by x co-ord
    '''
    in_points = sorted(in_points, key=lambda x:x[0])
    last_per = in_points[-1][0]
    prof = np.ones(last_per+1)

    per = 0
    for i, point in enumerate(in_points[1:]):

        x_diff = point[0] - in_points[i][0]
        y_diff = point[1] - in_points[i][1]
        grad =  y_diff / x_diff 

        prof[per:point[0]] = in_points[i][1] + np.arange(0,x_diff) * grad

        per = point[0]
    
    prof[-1]=in_points[-1][1]
    
    return prof

##_________________________________________________________________________##


def make_profile_shape(peak_spend, shed, _debug=False):
    '''Returns a profile (np.array) from input description of lifecycle shape
    (shed namedtuple with fields 'shed_name', 'uptake_dur', 'plat_dur', 'gen_mult',
    and peak spend.
    '''
    prof = np.array([float(shed.uptake_dur)] * (shed.uptake_dur+shed.plat_dur+1))
    prof[:shed.uptake_dur-1] = np.arange(1,shed.uptake_dur)
    prof[-1] = (shed.uptake_dur * shed.gen_mult)

    return prof * peak_spend / max(prof)


##_________________________________________________________________________##


def make_shapes(policy, flat=False, _debug=False):
    ''' DEPRECATED -  make_shapes1() better
    Helper function to generate arrays for plotting to visualise the shapes
    (rather than for further calculations)

    Input is a list or dict of SpendLine instances
    Returns the corresponding shapes, after aligning across launch delays
    '''
    if flat: policy = flatten(policy, _debug=_debug)

    min_delay = 0
    max_delay = 0

    out = []

    for s in policy:
        # make it work with a dict or a list

        if isinstance(policy, dict): s = policy[s]
        if s.launch_delay < min_delay: min_delay = s.launch_delay
        if s.launch_delay > max_delay: max_delay = s.launch_delay

    
    for s in policy:
        # make it work with a dict or a list
        name = None
        if isinstance(policy, dict): name = s
        elif isinstance(policy, list): name = s.name
        if isinstance(policy, dict): s = policy[s]        
        
        # assemble the elements
        spacer = s.launch_delay - min_delay
        main_phase = make_profile_shape(s.peak_spend, s.shed)

        # calculate any additional terminal growth period, so all spendlines line up
        tail = 12 + max_delay - s.launch_delay
        terminus = main_phase[-1] * ((1+s.term_gr)** np.arange(1,tail))
        

        ser = pd.Series(np.concatenate([np.zeros(spacer), main_phase, terminus]), name=name)
        # put together in a pd.Series
        out.append(ser)
    

    out_df = pd.DataFrame(out)
    try:
        out_df.index = pd.MultiIndex.from_tuples(out_df.index)
    except: pass
    
    return out_df.T

##_________________________________________________________________________##

def make_shape1(spendline=None, shed=None, z_pad=0, peak_spend=1, annualised=False, sav_rate=0, 
                net_spend=False, term_pad=1, term_gr=0, ser_out=True, name=None, _debug=False):
    '''Flexible function for generating shapes from sheds or spendlines.

    Everything but cohort growth - this is nonaccumulated
    Optionally add pads before and after (z_pad, term_pad) 
    Optionally return a pandas series or numpy array
    '''
    
    if _debug: print("\nIN FUNCTION:  ".ljust(20), inspect.stack()[0][3])
    if _debug: print("..called by:  ".ljust(20), inspect.stack()[1][3], end="\n\n")

    pad = 25

    ann_factor = 1
    if annualised:
        ann_factor = 144
        if_debug: print('annualising')
    elif _debug: print('not annualising')


    if spendline is not None:
        if _debug: print('using a passed spendline\n')
        shed = spendline.shed
        peak_spend = spendline.peak_spend
        sav_rate = spendline.sav_rate
        term_gr = spendline.term_gr
        z_pad = spendline.launch_delay

    elif shed is not None:
        if _debug: print('using a passed shed\n')

    else: print('need a spendline or a shed'); return 1

    zeros = np.zeros(z_pad)
    uptake = np.arange(1, shed.uptake_dur+1)
    plat = np.ones(shed.plat_dur) * shed.uptake_dur
    term = shed.gen_mult * (shed.uptake_dur * (1+term_gr) ** np.arange(1, term_pad+1))

    base = np.concatenate([zeros, uptake, plat, term])

    if _debug:
        print('shed: '.ljust(pad), shed)
        print('zpad passed: '.ljust(pad), z_pad)
        print('zeros arr: '.ljust(pad), zeros)
        print('uptake arr: '.ljust(pad), uptake)
        print('plat arr: '.ljust(pad), plat)
        print('term arr: '.ljust(pad), term)
        print('net spend: '.ljust(pad), net_spend)
        print('sav_rate: '.ljust(pad), sav_rate)
        print('\n-->base arr: '.ljust(pad))
        print(base, "\n")


    base = np.concatenate([zeros, uptake, plat, term])

    if not net_spend: 
        sav_rate=0
        if _debug: print('not netting, so rate is'.ljust(pad), sav_rate)

    else: 
        if _debug: print('netting, rate is'.ljust(pad), sav_rate)
    
    # do all the scaling.  Remembert uptake_dur is peak, due to the way
    # base assembled with np.arange for the uptake period
    scaling_factor = peak_spend * ann_factor * (1-sav_rate) / shed.uptake_dur

    base *= scaling_factor
    
    if _debug: 
        print('\nscaling factor'.ljust(pad), "\n", scaling_factor)
        print('\nbase after scaling'.ljust(pad), "\n", base)
  
    if ser_out:
        base = pd.Series(base, name=name)   

    if _debug: print("\nLEAVING FUNCTION:  ".ljust(20), inspect.stack()[0][3])
    if _debug: print("..returning to:  ".ljust(20), inspect.stack()[1][3], end="\n\n")

    return base


##_________________________________________________________________________##


def make_shapes1(scens, term_dur=0, start_m=None, flat=True, synch_start=False, 
                net_spend=True, multi_index=True, _debug=False):
    '''For an input dict of scenarios (scens), return a df of shapes,
    ensuring differential launch times are handled.
    
    term_dur    : minimum number of terminal periods to plot
    start_m     : optional start month, otherwise range index
    synch_start : option to move index start according to negative launch delays
    '''
    pad = 25
    if _debug: print("\nIN FUNCTION:  ".ljust(20), inspect.stack()[0][3])
    if _debug: print("..called by:  ".ljust(20), inspect.stack()[1][3], end="\n\n")

    if flat: scens = flatten(scens, _debug=False)

    pads = [25] + [8]*4
    out = pd.DataFrame()

    # work out the global dimensions: maximum overhang beyond zero, and max length of shed
    # for x in scens: 
    # if _debug: print("in line".ljust(pad), x)
    max_hang = min([scens[x].launch_delay for x in scens])
    max_len = max([(scens[x].shed.uptake_dur + scens[x].shed.plat_dur)  for x in scens])
    if _debug: print("max hang".ljust(pad), max_hang)
    if _debug: print("max len".ljust(pad), max_len)

    # use the global dimensions to construct consistent dimensions for each shape
    for x in scens:
        if _debug: print('launch_delay'.ljust(pads[0]), scens[x].launch_delay)
        shed_len = scens[x].shed.uptake_dur + scens[x].shed.plat_dur 
        z_pad = scens[x].launch_delay - max_hang
        term_pad = max_len + term_dur - scens[x].launch_delay - shed_len
        total = z_pad + shed_len + term_pad
        
        if _debug: 
            print('line'.ljust(pad), x)
            print('shed_len'.ljust(pad), shed_len)
            print('l_delay'.ljust(pad), scens[x].launch_delay)
            print('z_pad'.ljust(pad), z_pad)
            print('term_pad'.ljust(pad), term_pad)
            print('total'.ljust(pad), total)

        
        out[x] = make_shape1(shed = scens[x].shed, 
                               z_pad = z_pad, 
                               peak_spend = scens[x].peak_spend, 
                               sav_rate = scens[x].sav_rate,
                               net_spend = net_spend,
                               term_pad=term_pad, 
                               term_gr=scens[x].term_gr,
                               _debug = _debug)
                  

    if start_m is not None:
        if synch_start: 
            start_m = pd.Period(start_m, freq='M') + max_hang
            if _debug: print('new start m ', start_m)
        
        out.index = pd.PeriodIndex(freq='M', start = pd.Period(start_m, freq='M'), periods=total)   

    if multi_index: out.columns = pd.MultiIndex.from_tuples(out.columns)     

    if _debug: print("\LEAVING FUNCTION:  ".ljust(20), inspect.stack()[0][3])
    if _debug: print("..returning to:  ".ljust(20), inspect.stack()[1][3], end="\n\n")      

    return out


###_________________________________________________________________________##



def plot_diffs_ann_bar(start_m, n_pers, max_yrs=5,
                        policy=None, projs=None, diffs=None, net_spend=False,
                        fig=None, ax=None, figsize=None, legend=None, table=False,
                        return_fig=False, save_path=None):
    '''Makes a bar chart graph of the lifecycle shapes corresponding to the spendlines in a policy
    Can either generate a new plot, or add to existing axis (in which case pass ax)
    '''

    # get projections and diffs from policy if not passed
    if projs is None: 
        if policy is None and diffs is None: 
            print('need a policy or a set of projections or diffs')
        projs = project_policy(policy, start_m, n_pers, net_spend=net_spend)

    if diffs is None:
        diffs = projs.iloc[:,1:].subtract(projs.iloc[:,0], axis=0)    


    ind = projs.index.to_timestamp()

    annual_projs = projs.groupby(projs.index.year).sum().iloc[:max_yrs,:]
    annual_diffs = diffs.groupby(diffs.index.year).sum().iloc[:max_yrs,:]

    # set the name of the counterfactual
    col_zero = projs.columns[0]
    if isinstance(col_zero, tuple):
        counterfactual_name = col_zero[0]
    else: counterfactual_name = col_zero

    # create fig and ax, unless passed (which they will be if plotting in existing grid)
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    num_rects = len(annual_diffs.columns)
    rect_width = 0.5
    gap = 0.45
    for i, x in enumerate(annual_diffs):
        rect = ax.bar(annual_diffs.index + ((i/num_rects)*(1-gap)), annual_diffs[x], 
                        width=rect_width/num_rects) 
    
    title_str = ""
    if net_spend: title_str = " net"

    if len(annual_diffs.columns)==1:
        ax.set_title(("Annual{} spend impact, £m").format(title_str))
    else:
        ax.set_title(("Difference in annual{} spend vs " + counterfactual_name +", £m").format(title_str))
    ax.tick_params(axis='x', bottom='off')
    ax.grid(False, axis='x')
    # for t in ax.get_xticklabels():
    #     t.set_rotation(45)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()])
    ax.legend(annual_diffs.columns)
    if len(annual_diffs.columns)>2:  ax.legend(annual_diffs.columns)


    if table:
        ax.set_xticks([])

        rows = []
        for x in annual_diffs:
            rows.append(["{:0.2f}".format(y) for y in annual_diffs[x]])

        c_labels = list(annual_diffs.index)
        tab = ax.table(cellText=rows, colLabels=c_labels, rowLabels= ['    £m  '])
        tab.set_fontsize(12)
        tab.scale(1,2)

    if save_path is not None:
        fig.savefig(save_path)

    if return_fig: return(fig)

##_________________________________________________________________________##




def plot_shapes_line(policy, annualise=True, fig=None, ax=None, figsize=None, return_fig=False, save_path=None):
    '''Makes a simple line graph of the lifecycle shapes corresponding to the spendlines in a policy
    '''
    shapes = make_shapes(policy)

    if annualise:
        shapes *= 12*12

    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    for i, s in enumerate(shapes):
        # if isinstance(policy, dict): s = policy[s] # make it work with dict
        
        if i == 0: 
            ax.plot(np.arange(len(shapes[s]))/12, shapes[s], color='black')
        else:
            ax.plot(np.arange(len(shapes[s]))/12, shapes[s], alpha=0.75)

        ax.set_title("Lifecycles")
        ax.set_xlabel('years post launch')
        ax.set_ylabel('£m, annualised')
        ax.legend(shapes.columns)

    if save_path is not None:
        fig.savefig(save_path)

    if return_fig: return(fig)



##_________________________________________________________________________##


def project_policy(policy, start_m='1-2019', n_pers=120, shapes=None,
                    synch_start=False, diffs_out=False, annual=False, net_spend=True,
                    nans_to_zero=True, multi_index=True, _debug=False):  
    '''For a list of spendlines (with a start month and number of periods),
    returns a df with projections.
    
    The function orients individual  projections according to the launch delays
    across the whole set of spendlines, inserting spacers as necessary.

    Does not apply any savings assumptions - these need to happen before or after.

    Note this can't work on just shapes - needs terminal growth and coh growth
    
    PARAMETERS
    
    policy     : a list of spendlines (see SpendLine class)
    
    start_m    : start month for the projections 
                    (i.e. period zero for the spendline with lowest launch delay)
               : format eg '3-2019' is march 2019 (pd.Period standard)
    synch_start:
               : if the spendlines have a negative minimum launch delay
               :  (i.e. if one involves bringing launch forward) then move the 
               :  actual start month forward from the input.
               :  This is like saying that start_m is actually period zero, but 
               :  earlier periods are possible.  This lets different sets of policies be
               :  oriented correctly in time.
               : Note this goes through make_shapes1
              
    n_pers     : number of periods to project.  Normally months
    
    '''
    pad = 25
    if _debug: print("\nIN FUNCTION:  ".ljust(20), inspect.stack()[0][3])
    if _debug: print("..called by:  ".ljust(20), inspect.stack()[1][3], end="\n\n")

    out = pd.DataFrame()

    policy = flatten(policy, _debug=_debug)
    if _debug: print('flattened keys'.ljust(pad), policy.keys())

    # need to work out the max overhang.  NB this is also done in make_shapes1
    max_hang = min([policy[x].launch_delay for x in policy])
    if _debug: print("max hang is".ljust(pad), max_hang)


    # get the shapes - keys will be same as col heads
    # NB MUST PASS NET SPEND FLAG.  The policy carries the sav rates
    shapes = make_shapes1(policy, term_dur=0, start_m=start_m, net_spend=True,
                        flat=True, multi_index=False, synch_start=synch_start, _debug=_debug)
    if _debug: print('shapes returned:\n', shapes.head())

    # work out new time frame, given any extension for negative launch delay
    # len_reqd = n_pers - max_hang
    # if _debug: print('new timeframe'.ljust(pad), len_reqd)

    # can now iterate through shapes, and cross ref the spendline
    for s in policy:
        if _debug: print('column: '.ljust(pad), s)
        out[s] = pf.get_forecast1(shapes[s], term_gr = policy[s].term_gr,
                                             coh_gr = policy[s].coh_gr,  
                                             l_stop = policy[s].launch_stop,
                                             n_pers = n_pers, 
                                             _debug=_debug)

    if _debug: print('length returned'.ljust(pad), len(out))

    # the index may be longer than the n pers, because of any overhangs erc
    ind = pd.PeriodIndex(start=start_m, freq='M', periods=len(out.index))
    out.index = ind

    if multi_index: out.columns=pd.MultiIndex.from_tuples(out.columns)

    if diffs_out:
        out = out.iloc[:,1:].subtract(out.iloc[:,0], axis=0)

    if annual:
        out = out.groupby(out.index.year).sum()

    if nans_to_zero: out[out.isnull()] = 0

    if _debug: print("\LEAVING FUNCTION:  ".ljust(20), inspect.stack()[0][3])
    if _debug: print("..returning to:  ".ljust(20), inspect.stack()[1][3], end="\n\n")

    return out


##_________________________________________________________________________##


def projection(shapes, start_m='1-2019', n_pers=120, synch_start=True, _debug=False):
    '''Revised function to get projections requiring an input shapes df
    '''
    pad = 25
    if _debug: print("\nIN FUNCTION:  ".ljust(20), inspect.stack()[0][3])
    if _debug: print("..called by:  ".ljust(20), inspect.stack()[1][3], end="\n\n")

    out = pd.DataFrame()

    for s in shapes:
        pf.get_forecast1()

##_________________________________________________________________________##


class Shed(namedtuple('shed', 'shed_name uptake_dur plat_dur gen_mult')):
    '''The parameters to define a linear 'shed'-like lifecycle profile, 
    in a named tuple.
    
    .shed_name  : the name of the shed shape
    .uptake_dur : the number of periods of linear growth following launch
    .plat_dur   : the number of periods of constant spend following uptake
    .gen_mult   : the change on patent expiry (multiplier)
    '''
    pass


##_________________________________________________________________________##


class SpendLine():
    '''Stores information required to define a policy spendline.  I

    CONSTRUCTOR ARGUMENTS

    name            : the name of the spendline [REQD]

    shed            : namedtuple describing the lifecycle profile [REQD]
     .shed_name      : the name of the shed profile
     .uptake_dur     : duration of (linear) uptake period
     .plat_dur       : duration of (flat) plateau
     .gen_mult       : impact of patent expiry (multiplier, eg 0.2)
    
    term_gr         : the rate of growth to be applied (indefinitely) 
                    :  after the drop on patent expiry

    launch_delay    : negative if launch brought fwd
    coh_gr          : cohort growth rate
    peak_spend      : peak spend to which profile is normalised
    icer            : the incremental cost-effectiveness ratio (£cost/QALY)
    sav_rate        : proportion of spend that substitute other drug spend

    log             : general dict for holding whatever

    All time units are general - usually implied to be months.  
    But no annualisation within this class except in the string.

    '''
    def __init__(self, name, shed, 
                       term_gr=0, launch_delay=0, launch_stop=None, coh_gr=0, 
                       peak_spend=1, icer=None, sav_rate=0,
                       log=None):

        self.name = name
        self.shed = shed
        self.term_gr = term_gr
        self.coh_gr = coh_gr
        self.peak_spend = peak_spend
        self.launch_delay = launch_delay
        self.launch_stop = launch_stop
        self.icer = icer
        self.sav_rate = sav_rate

        # create a general purpose dict for holding whatever
        if log==None: 
            self.log = {}
        else: self.log = log

    
    def __str__(self):
        pad1 = 35
        pad2 = 10
        return "\n".join([
            "SpendLine name:".ljust(pad1) + str(self.name).rjust(pad2),
            "peak spend pm of monthly cohort:".ljust(pad1) + "{:0,.2f}".format(self.peak_spend).rjust(pad2),
            "peak spend pa of annual cohort:".ljust(pad1) + "{:0,.2f}".format(self.peak_spend*12*12).rjust(pad2),
            "ICER, £/QALY:".ljust(pad1) + "{:0,.0f}".format(self.icer).rjust(pad2),
            "savings rate:".ljust(pad1) + "{:0,.1f}%".format(self.sav_rate*100).rjust(pad2),
            "shed:".ljust(pad1),
            "  - shed name:".ljust(pad1) + self.shed.shed_name.rjust(pad2),
            "  - uptake_dur:".ljust(pad1) + str(self.shed.uptake_dur).rjust(pad2),
            "  - plat_dur:".ljust(pad1) + str(self.shed.plat_dur).rjust(pad2),
            "  - gen_mult:".ljust(pad1) + str(self.shed.gen_mult).rjust(pad2),
            "term_gr:".ljust(pad1) + str(self.term_gr).rjust(pad2),
            "launch_delay:".ljust(pad1) + str(self.launch_delay).rjust(pad2),
            "launch_stop:".ljust(pad1) + str(self.launch_stop).rjust(pad2),
            "coh_gr:".ljust(pad1) + str(self.coh_gr).rjust(pad2),
            "log keys:".ljust(pad1) + ", ".join(self.log.keys()).rjust(pad2),
            "\n\n"

        ])
 

    def __repr__(self):
        return self.__str__()

    def get_shape(self):
        return make_profile_shape(self.peak_spend, self.shed)

