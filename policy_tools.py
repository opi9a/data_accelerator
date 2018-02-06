import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
import matplotlib.ticker as ticker
import projection_funcs as pf
from collections import namedtuple


def cost_ratio(sav_rate, k1=None, k2=None, ratio=None):
    '''Returns the cost ratio (multiplier) to apply to baseline costs 
    to reflect a change in threshold (k1->k2), given the savings (drugs substitution)
    as a proportion r of the original costs

    k1 = (C1 - s)/dQ : k is cost/QALY, c is cost of drug, s is savings
    r = sC1          : r is savings as proportion of initial costs
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



def dump_to_xls(res_df, outfile, policy=None, scenario=None):
    '''Dump a results DataFrame to an xls, with option to make shapes from a list 
    or dict of scenarios (policy), or from a scenario alone
    '''
    writer = pd.ExcelWriter(outfile)
  
    if policy is not None:

        # find if it's one or two layers (have to do for list and dict different)
        if isinstance(policy, dict):
            layer1 = policy[list(policy.keys())[0]]

        elif isinstance(policy, list):
            layer1 = policy[0]

        if not type(layer1) in [list, dict]:
            policy = [policy]

        params_table = make_params_table(policy)
        summary_table=None
        try:
            summary_table = params_table.append(res_df.groupby(res_df.index.year).sum())
            summary_table.to_excel(writer, 'summary')
        except:
            params_table.to_excel(writer, 'params')
        dump_shapes(policy).to_excel(writer, 'shapes')

    if scenario is not None:
        params_table = make_params_table(scenario)
        summary_table=None
        try:
            summary_table = params_table.append(res_df.groupby(res_df.index.year).sum())
            summary_table.to_excel(writer, 'summary')
        except:
            params_table.to_excel(writer, 'params')
        make_shapes(scenario).to_excel(writer, 'shapes')        

    
    res_df.to_excel(writer, 'main_res')

    # if more than one column, do a stack
    if len(res_df.columns)>1:
    # excel writer doesn't like pd.Periods it seems
        new_ind = res_df.index.to_timestamp()
        stacked = res_df.copy().set_index(new_ind)
        
        #need to check how many levels to work out how many times to unstack
        col_depth = len(res_df.columns.names)

        for d in range(col_depth):
            stacked = stacked.stack()

        stacked.rename('spend')
        new_names = ['Date', 'SpendLine', 'Scenario']
        stacked.index.names = new_names[:col_depth+1]
        stacked.to_excel(writer, 'stacked')
    
    writer.save()


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


def impute_peak_sales(df_slice, cutoff, shed, coh_growth_pa, term_growth_pa, ave_interval=3, 
                        _debug=False):
    '''For an input df slice (NB not by launch date), and lifecycle shape, 
    returns the impled peak sales associated with the lifecycle shape
    (i.e. for an individual product or cohort).
    
    Returns peak sales per period for one period's worth of launches.  
    So if period is month, multiply *12*12 for peak sales pa of a year of launches.
    
    ave_interval is the interval over which terminal sales in the observed set is calculated

    Works by comparing a synthetic cumulation based on the lifecycle shape with actual sales, 
    initial ('stub') sales since a cutoff date.

    First make the synthetic cumulation, scaled to peak sales = 1.
    Then align the initial period with the start date of the stub.
    Then get the actual sales at the end of the observed stub 'end_sales'.
    Compare this to the sales at the corresponding period of the synthetic cumulation
    The ratio of these is the implied lifecycle peak sales.
    '''
        
    # make sure cutoff is a period (will take a string here)
    cutoff = pd.Period(cutoff, freq='M')
    
    # work out how many periods since cutoff
    n_pers = df_slice.columns[-1] - cutoff
    if _debug: print('npers', n_pers)
    
    # get last real observations - only need sales in last period really
    ixs = pf.get_ix_slice(df_slice, dict(start_month=slice(cutoff,None,None)))
    summed = df_slice.loc[ixs,:].sum()
    
    if _debug: print('summed df tail\n', summed.tail())

    end_sales = summed[-ave_interval:].mean()
    if _debug: print('end_sales {:0,.0f}'.format(end_sales))

    # make unscaled simulated cumulation
    shape = make_profile_shape(1, shed)
    if _debug: print("shape\n", shape)

    sim = pf.get_forecast(shape, l_start=0, l_stop=n_pers, 
                          coh_growth=coh_growth_pa/12, term_growth=term_growth_pa/12)

    if _debug: print('tail of cumulation\n', sim.tail())

    end_sim = sim[-ave_interval:].mean()

    if _debug: print('end_sim tail', end_sim)

    return end_sales / end_sim

##_________________________________________________________________________##


def make_params_table(pol_dict, index=None):

    if index is None:
        index = 'peak_sales_pa sav_rate uptake_dur plat_dur gen_mult term_gr coh_gr'.split()

    df = pd.DataFrame(index=index)

    # make an internal func, so this works with 1 or 2 layer dicts
    def _get_stuff(low_dict, parent=None):
        # sub_df=pd.DataFrame(index=index)
        for q in low_dict:
            params = [low_dict[q].peak_sales_pm*12*12, # double annualised
                     low_dict[q].sav_rate,
                     int(low_dict[q].shed.uptake_dur),
                     int(low_dict[q].shed.plat_dur),
                     low_dict[q].shed.gen_mult,
                     low_dict[q].terminal_gr,
                     low_dict[q].cohort_gr]

            # if doing this in a 2 layer dict, need to get a tuple column name
            if parent is None:
                col_name = q
            else: 
                col_name = (parent, q)
            df[col_name] = params


    # case if 1 layer dict(i.e. a single scenario)

    # find if it's one or two layers (have to do for list and dict different)
    if isinstance(pol_dict, dict):
        layer1 = pol_dict[list(pol_dict.keys())[0]]

    elif isinstance(pol_dict, list):
        layer1 = pol_dict[0]

    if not type(layer1) in [list, dict]:
        _get_stuff(pol_dict)

    else:
        for p in pol_dict:
            if isinstance(pol_dict, dict):
                _get_stuff(pol_dict[p], p)

            elif isinstance(pol_dict, list):
                _get_stuff(p)

    try:
        df.columns = pd.MultiIndex.from_tuples(df.columns)
    except: pass
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


def make_profile_shape(peak_sales_pm, shed, _debug=False):
    '''Returns a profile (np.array) from input description of lifecycle shape
    (shed namedtuple with fields 'shed_name', 'uptake_dur', 'plat_dur', 'gen_mult',
    and peak sales.
    '''
    prof = np.array([float(shed.uptake_dur)] * (shed.uptake_dur+shed.plat_dur+1))
    prof[:shed.uptake_dur-1] = np.arange(1,shed.uptake_dur)
    prof[-1] = (shed.uptake_dur * shed.gen_mult)

    return prof * peak_sales_pm / max(prof)


##_________________________________________________________________________##


def make_shapes(policy, flat=False, _debug=False):
    '''Helper function to generate arrays for plotting to visualise the shapes
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
        main_phase = make_profile_shape(s.peak_sales_pm, s.shed)

        # calculate any additional terminal growth period, so all spendlines line up
        tail = 12 + max_delay - s.launch_delay
        terminus = main_phase[-1] * ((1+s.terminal_gr)** np.arange(1,tail))
        

        ser = pd.Series(np.concatenate([np.zeros(spacer), main_phase, terminus]), name=name)
        # put together in a pd.Series
        out.append(ser)
    

    out_df = pd.DataFrame(out)
    try:
        out_df.index = pd.MultiIndex.from_tuples(out_df.index)
    except: pass
    
    return out_df.T

##_________________________________________________________________________##

def make_shape1(shed, z_pad=0, peak_sales_pm=1, annualised=True, sav_rate=0, net_spend=False, 
                term_pad=0, term_gr=0, ser_out=True, name=None, _debug=False):
    '''Generate 
    '''
    
    base = np.concatenate([np.zeros(z_pad), 
                    np.arange(1, shed.uptake_dur+1), 
                    np.ones(shed.plat_dur) * shed.uptake_dur,
                    shed.gen_mult * shed.uptake_dur * (1+term_gr) ** np.arange(1, term_pad+1)])

    if not net_spend: sav_rate=0
    
    base *= peak_sales_pm * (1-sav_rate) / shed.uptake_dur
    
    if ser_out:
        base = pd.Series(base, name=name)   

    if annualised: base *=12*12     

    return  base


##_________________________________________________________________________##


def make_shapes1(pol, term_dur=12, start_m=None, flat=False, synch_start=False, 
            multi_index=False, _debug=False):
    '''For an input dict of scenarios (pol), return a df of shapes.
    
    term_dur    : minimum number of terminal periods to plot
    start_m     : optional start month, otherwise range index
    synch_start : option to move index start according to negative launch delays
    '''

    if flat: pol = flatten(pol, _debug=_debug)

    pads = [10] + [8]*4
    out = pd.DataFrame()

    # work out the global dimensions: maximum overhang beyond zero, and max length of shed
    for x in pol: 
        if _debug: print(x)
        max_hang = min([pol[x].launch_delay for x in pol])
        max_len = max([(pol[x].shed.uptake_dur + pol[x].shed.plat_dur)  for x in pol])
    
    if _debug: print("line".ljust(pads[0]), 
                      "shed_len".rjust(pads[1]), 
                      "zpad".rjust(pads[2]), 
                      "t pad".rjust(pads[3]),
                      "total".rjust(pads[4]))
           
    # use the global dimensions to construct consistent dimensions for each shape
    for x in pol:
        shed_len = pol[x].shed.uptake_dur + pol[x].shed.plat_dur 
        z_pad = pol[x].launch_delay - max_hang
        term_pad = max_len + term_dur - pol[x].launch_delay - shed_len
        total = z_pad + shed_len + term_pad
        
        if _debug: print(str(x).ljust(pads[0]), 
                          str(shed_len).rjust(pads[1]), 
                          str(z_pad).rjust(pads[2]), 
                          str(term_pad).rjust(pads[3]),
                          str(total).rjust(pads[4]))
        
        out[x] = make_shape1(pol[x].shed, 
                               z_pad = z_pad, 
                               peak_sales_pm = pol[x].peak_sales_pm, 
#                                sav_rate = pol[x].sav_rate,
                               term_pad=term_pad, 
                               term_gr=pol[x].terminal_gr,
                               _debug = _debug)
                  

    if start_m is not None:
        if synch_start: 
            start_m = pd.Period(start_m, freq='M') + max_hang
            if _debug: print('new start m ', start_m)
        
        out.index = pd.PeriodIndex(freq='M', start = pd.Period(start_m, freq='M'), periods=total)   

    if multi_index: out.columns = pd.MultiIndex.from_tuples(out.columns)         
            
    return out

##_________________________________________________________________________##


def plot_ann_diffs(projs, max_yrs=5, fig=None, ax=None, figsize=None, 
                    table=False, legend=None, net_spend=False, return_fig=False, save_path=None):
    '''Plots a bar chart of annual data, subtracting the first column
    Can either generate a new plot, or add to existing axis (in which case pass ax)
    '''


    diffs = projs.iloc[:,1:].subtract(projs.iloc[:,0], axis=0)
    diffs = diffs.groupby(diffs.index.year).sum().iloc[:max_yrs,:]

    ind = diffs.index


    # set the name of the counterfactual
    col_zero = projs.columns[0]
    if isinstance(col_zero, tuple):
        counterfactual_name = col_zero[0]
    else: counterfactual_name = col_zero

    # create fig and ax, unless passed (which they will be if plotting in existing grid)
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    num_rects = len(diffs.columns)
    rect_width = 0.5
    gap = 0.45
    for i, x in enumerate(diffs):
        rect = ax.bar(diffs.index + ((i/num_rects)*(1-gap)), diffs[x], 
                        width=rect_width/num_rects) 
    title_str = ""
    if net_spend: title_str = " net"

    ax.set_title("Difference in{} annual spend vs ".format(title_str) + counterfactual_name +", £m")
    ax.tick_params(axis='x', bottom='off')
    ax.grid(False, axis='x')
    # for t in ax.get_xticklabels():
    #     t.set_rotation(45)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()])
    
    if legend is not None:
        ax.legend(legend)

    else:
        ax.legend(diffs.columns)
        if len(diffs.columns)>2:  ax.legend(diffs.columns)

    if table:
        ax.set_xticks([])

        rows = []
        for x in diffs:
            rows.append(["{:0.2f}".format(y) for y in diffs[x]])

        row_labs = None
        if legend: row_labs = legend
        else: row_labs = diffs.columns

        c_labels = list(diffs.index)
        tab = ax.table(cellText=rows, colLabels=c_labels, rowLabels= row_labs)
        tab.set_fontsize(12)
        tab.scale(1,2)
        tab.auto_set_font_size

    if save_path is not None:
        fig.savefig(save_path)

    if return_fig: return(fig)

##_________________________________________________________________________##


def plot_cumspend_line(start_m=None, n_pers=None, 
                        policy=None, projs=None, annualise=True, net_spend=False, plot_pers=None,
                        fig=None, ax=None, figsize=None, return_fig=False, save_path=None):
    '''Plots a  line graph of the cumulated spend corresponding to the spendlines in a policy

    Can either generate a new plot, or add to existing axis (in which case pass ax)

    Can either generate projections and index from the policy, or use existing if passed
    '''

    # get projections from policy if not passed
    if projs is None: 
        projs = project_policy(policy, start_m, n_pers, net_spend=net_spend)

    if annualise: projs *= 12

    if plot_pers is not None:
        projs = projs.iloc[:plot_pers,:]

    ind = projs.index.to_timestamp()

    # create fig and ax, unless passed (which they will be if plotting in existing grid)
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    for i, p in enumerate(projs):
        if i==0: 
            ax.plot(ind, projs[p].values, color='black') 
        else:
            ax.plot(ind, projs[p].values, alpha=0.75)  

    for t in ax.get_xticklabels():
        t.set_rotation(45)

    ax.legend(['1', '2'])
    ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()])

    title_str = ""
    if net_spend: title_str = " net"
   
    ax.set_title("Accumulated{} spend".format(title_str))
    ax.legend(projs.columns)  

    if save_path is not None:
        fig.savefig(save_path)

    if return_fig: return(fig)

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


def plot_impact_grid3(policy, start_m, n_pers, projs=None, diffs=None, max_bar_yrs=5, plot_pers=None, net_spend=False,
                        save_path=None, plot_bar=True, return_fig=False,
                        table=False):
    '''Plots a grid of charts.

    Going to change this to use individual plotting functions 
    for each chart commonly needed, so can then choose whatever grid layout
    '''
    
    if projs is None: projs = project_policy(policy, start_m, n_pers, net_spend=net_spend)
    ind = projs.index.to_timestamp()
    if diffs is None: diffs = projs.iloc[:,1:].subtract(projs.iloc[:,0], axis=0)    

# plot all shapes and cumulated projections
# for diffs, calc vs first columnb
    
    annual_projs = projs.groupby(projs.index.year).sum()
    annual_diffs = diffs.groupby(diffs.index.year).sum()

    tab_rows = 2
    if plot_bar:
        tab_rows +=1
        if table:
            tab_rows +=1

    fig = plt.figure(figsize=(12,tab_rows*5))
    rcParams['axes.titlepad'] = 12

    ax0 = plt.subplot2grid((tab_rows,2), (0, 0))
    plot_shapes_line(policy, annualise=True, ax=ax0)

    ax1 = plt.subplot2grid((tab_rows,2), (0, 1))
    plot_cumspend_line(start_m=start_m, n_pers=n_pers, annualise=True, plot_pers=plot_pers, policy=policy, net_spend=net_spend, ax=ax1)

    if plot_bar:
        ax2 = plt.subplot2grid((tab_rows,2), (1, 0), colspan=2)
        plot_diffs_ann_bar(start_m=start_m, n_pers=n_pers, ax=ax2, projs=projs, diffs=diffs, 
                    table=True, max_yrs=max_bar_yrs, net_spend=net_spend)

    # if table:
    #     tab = plt.subplot2grid((tab_rows,2), (2, 0), colspan=2)
    #     tab.set_frame_on(False)
    #     tab.set_xticks([])
    #     tab.set_yticks([])

    #     rowvals = ["{:0,.0f}".format(x) for x in annual_diffs.iloc[:,0].values]
    #     the_table = tab.table(cellText=[rowvals], rowLabels=['spend, £m'],
    #                         loc='top')
    #     the_table.auto_set_font_size(False)
    #     the_table.set_fontsize(10)

    # fig.text(0.13,0.8,'here is text')

    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    if save_path is not None:
        fig.savefig(save_path)

    if return_fig:
        return fig

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


def project_policy(policy, start_m='1-2019', n_pers=120, 
                    synch_start=True, net_spend=False, diffs_out=False, annual=False):  
    '''For a list of spendlines (with a start month and number of periods),
    returns a df with projections.
    
    The function orients individual  projections according to the launch delays
    across the whole set of spendlines, inserting spacers as necessary.
    
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

    net_spend  : return spend net of savings (sav_rate of SpendLine)
               
    n_pers     : number of periods to project.  Normally months
    
    '''
    min_delay = 0
    out = []

    # find the minimum launch delay - NB likely negative if any effects
    for s in policy:
        # make it work with a dict or a list
        if isinstance(policy, dict): s = policy[s]
        if s.launch_delay < min_delay: min_delay = s.launch_delay
   

    # reset start date 
    if synch_start:
        start_m = pd.Period(start_m, freq='M') + min_delay

    for s in policy:
        if isinstance(policy, dict): s = policy[s]
        spacer = s.launch_delay - min_delay
        spaced_shape = np.concatenate([np.zeros(spacer), s.get_shape()])
        spend_out = pd.Series(pf.get_forecast(spaced_shape, coh_growth=s.cohort_gr/12, 
                                                term_growth=s.terminal_gr/12, 
                                                l_start=0, l_stop=n_pers, 
                                                name=s.name))
        if net_spend:
            spend_out *= (1-s.sav_rate)
        out.append(spend_out)

    ind = pd.PeriodIndex(start=start_m, freq='M', periods=n_pers)
    df = pd.DataFrame(out).T
    df.index = ind

    if diffs_out:
        df = df.iloc[:,1:].subtract(df.iloc[:,0], axis=0)

    if annual:
        df = df.groupby(df.index.year).sum()

    return df




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

    name            : the name of the spendline 

    shed            : namedtuple describing the lifecycle profile
     .shed_name      : the name of the shed profile
     .uptake_dur     : duration of (linear) uptake period
     .plat_dur       : duration of (flat) plateau
     .gen_mult       : impact of patent expiry (multiplier, eg 0.2)
    
    terminal_gr     : the rate of growth to be applied (indefinitely) 
                    : after the drop on patent expiry

    launch_delay    : negative if launch brought fwd
    cohort_gr       : cohort growth rate
    peak_sales_pa      : peak sales to which profile is normalised

    '''
    def __init__(self, name, shed, terminal_gr=0, launch_delay=0, cohort_gr=0, peak_sales_pm=1, sav_rate=0):
        self.name = name
        self.shed = shed
        self.terminal_gr = terminal_gr
        self.cohort_gr = cohort_gr
        self.peak_sales_pm = peak_sales_pm
        self.launch_delay = launch_delay
        self.sav_rate = sav_rate

    
    def __str__(self):
        pad1 = 35
        pad2 = 10
        return "\n".join([
            "SpendLine name:".ljust(pad1) + str(self.name).rjust(pad2),
            "peak sales pm of monthly cohort:".ljust(pad1) + "{:0,.2f}".format(self.peak_sales_pm).rjust(pad2),
            "peak sales pa of annual cohort:".ljust(pad1) + "{:0,.2f}".format(self.peak_sales_pm*12*12).rjust(pad2),
            "savings rate:".ljust(pad1) + "{:0,.1f}%".format(self.sav_rate*100).rjust(pad2),
            "shed:".ljust(pad1),
            "  - shed name:".ljust(pad1) + self.shed.shed_name.rjust(pad2),
            "  - uptake_dur:".ljust(pad1) + str(self.shed.uptake_dur).rjust(pad2),
            "  - plat_dur:".ljust(pad1) + str(self.shed.plat_dur).rjust(pad2),
            "  - gen_mult:".ljust(pad1) + str(self.shed.gen_mult).rjust(pad2),
            "terminal_gr:".ljust(pad1) + str(self.terminal_gr).rjust(pad2),
            "launch_delay:".ljust(pad1) + str(self.launch_delay).rjust(pad2),
            "cohort_gr:".ljust(pad1) + str(self.cohort_gr).rjust(pad2),
            "\n\n"

        ])
 

    def __repr__(self):
        return self.__str__()

    def get_shape(self):
        return make_profile_shape(self.peak_sales_pm, self.shed)

