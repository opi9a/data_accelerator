import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('bmh')

import RuleSet
import r_funcs as rf

def make_rsets(df, params_dict, 
                xtrap=False, return_sum = False, return_setting_sums=False,
                trim=False, ann_sum=False):
    '''Helper function to make rulesets objects based on input parameters

    Default just returns a dict of rulesets with the params added.

    Setting `xtrap` flag will return dict of rulesets populated with results of calling xtrap on each
     - i.e. it will actually do the extrapolation functions and assign to .past, .fut, .joined and .sum

    Setting `return_sum` or `return_setting_sums` flags will return summed datasets as expected.

    Setting `trim` with a tuple or list (start, end) will trim the output sum dfs
    '''

    cut_off = params_dict['cut_off']
    biol_shed = params_dict['biol_shed']
    non_biol_shed = params_dict['non_biol_shed']
    loe_delay = params_dict['loe_delay']
    biol_term_gr = params_dict['biol_term_gr']
    non_biol_term_gr = params_dict['non_biol_term_gr']
    biol_coh_gr = params_dict['biol_coh_gr']
    non_biol_coh_gr = params_dict['non_biol_coh_gr']
    n_pers = params_dict['n_pers']

    rsets = {}
    # existing product rulesets - NB MUST SET CUTOFF TO MINUS 1 TO AVOID OVERLAP
    rsets['biol_sec'] = RuleSet.RuleSet(df, name='biol_sec', 
                               index_slice = dict(biol=True, setting='secondary', start_month=slice(None, cut_off-1, None)),
                               func = rf.r_trend,
                               f_args = dict(shed=biol_shed, term_gr=biol_term_gr, loe_delay=loe_delay))

    rsets['nonbiol_sec'] = RuleSet.RuleSet(df, name='nonbiol_sec', 
                               index_slice = dict(biol=False, setting='secondary', start_month=slice(None, cut_off-1, None)),
                               func = rf.r_trend,
                               f_args = dict(shed=non_biol_shed, term_gr=non_biol_term_gr, loe_delay=loe_delay))

    rsets['biol_prim'] = RuleSet.RuleSet(df, name='biol_prim', 
                               index_slice = dict(biol=True, setting='primary', start_month=slice(None, cut_off-1, None)),
                               func = rf.r_trend,
                               f_args = dict(shed=biol_shed, term_gr=biol_term_gr, loe_delay=loe_delay))

    rsets['nonbiol_prim'] = RuleSet.RuleSet(df, name='nonbiol_prim', 
                               index_slice = dict(biol=False, setting='primary', start_month=slice(None, cut_off-1, None)),
                               func = rf.r_trend,
                               f_args = dict(shed=non_biol_shed, term_gr=non_biol_term_gr, loe_delay=loe_delay))

    # future launches rulesets
    rsets['biol_sec_fut'] = RuleSet.RuleSet(df, name='biol_sec_fut', 
                               index_slice = dict(biol=True, setting='secondary', start_month=slice(cut_off, None, None)),
                               func = rf.r_fut_tr,
                               f_args = dict(shed=biol_shed, loe_delay=loe_delay, term_gr=biol_term_gr, coh_gr=biol_coh_gr/12, cut_off=cut_off))

    rsets['nonbiol_sec_fut'] = RuleSet.RuleSet(df, name='nonbiol_sec_fut', 
                               index_slice = dict(biol=False, setting='secondary', start_month=slice(cut_off, None, None)),
                               func = rf.r_fut_tr,
                               f_args = dict(shed=non_biol_shed, loe_delay=loe_delay, term_gr=non_biol_term_gr, coh_gr=non_biol_coh_gr/12, cut_off=cut_off))

    rsets['biol_prim_fut'] = RuleSet.RuleSet(df, name='biol_prim_fut', 
                               index_slice = dict(biol=True, setting='primary', start_month=slice(cut_off, None, None)),
                               func = rf.r_fut_tr,
                               f_args = dict(shed=biol_shed, loe_delay=loe_delay, term_gr=biol_term_gr, coh_gr=biol_coh_gr/12, cut_off=cut_off))

    rsets['nonbiol_prim_fut'] = RuleSet.RuleSet(df, name='nonbiol_prim_fut', 
                               index_slice = dict(biol=False, setting='primary', start_month=slice(cut_off, None, None)),
                               func = rf.r_fut_tr,
                               f_args = dict(shed=non_biol_shed, loe_delay=loe_delay, term_gr=non_biol_term_gr, coh_gr=non_biol_coh_gr/12, cut_off=cut_off))
    
    if xtrap or return_sum or return_setting_sums: 
        for r in rsets:
            rsets[r].xtrap(n_pers)

    if return_sum or return_setting_sums:
        sums = pd.concat([rsets[x].summed for x in rsets], axis=1)
        if trim:
            sums = sums.loc[slice(pd.Period(trim[0], freq='M'), pd.Period(trim[1], freq='M'),None),:]
        if ann_sum:
            sums = sums.groupby(sums.index.year).sum(axis=1)

    if return_sum: 
        if return_setting_sums: 
            print('Returning single sum only - to return sums by setting you need to turn off the `return_sum` flag')
        return sums

    elif return_setting_sums:
        by_setting = sums.groupby(lambda x: 'sec' in x, axis=1).sum()
        by_setting.columns = ['Primary', 'Secondary']
        return by_setting

    else: return rsets


##____________________________________________________________________________________________________________##


def plot_rset_projs(rs_dict, num_plots=12, n_pers=120, out_folder='figs/test_plot_rset/', 
                    save_fig=True, zeros_to_nans=False, _debug=False):
    
    '''Helper function for plotting projections of individual products in rulesets.

    For a dict of rulesets, sorts by max sales and takes num_plots highest.

    Then for each generates a plot containing each product in the high sales set.
    '''

    plt.close("all")
    pad = 35

    # iterate non future rsets
    for r in [x for x in rs_dict.keys() if not x.endswith('_fut')]:
        # get the top n by max sales
        print('\nin rset'.ljust(pad), r)
        top_n = rs_dict[r].past.loc[rs_dict[r].past.max(axis=1).sort_values(ascending=False).head(num_plots).index]
        
        #get into £m annualisex
        top_n *=12/10**8
                      
        # get the LIST OF dfs from r_trend with an _out_type='df' flag
        df_list = rf.r_trend(top_n, n_pers=n_pers,
                                   shed = rs_dict[r].f_args['shed'],
                                   term_gr = rs_dict[r].f_args['term_gr'],
                                   loe_delay = rs_dict[r].f_args['loe_delay'],
                                   _out_type='df')
        
        eff_plat_len = rs_dict[r].f_args['shed'].plat_dur + rs_dict[r].f_args['loe_delay']
        eff_total_len = rs_dict[r].f_args['shed'].uptake_dur + rs_dict[r].f_args['shed'].plat_dur + rs_dict[r].f_args['loe_delay']
        if _debug: print('eff plat length is ', eff_plat_len)
        if _debug: print('eff total length is ', eff_total_len)
        
        # make graph for this ruleset (remember in a loop here already)
        fig, ax = plt.subplots(num_plots, figsize=(12, num_plots * 6))

        # now loop through the returned dataframes - one per product
        for i, df in enumerate(df_list):

            if _debug: print('\ndf number'.ljust(pad), i)
            if _debug: print('..corresponding product name'.ljust(pad), top_n.iloc[i].name[0])

            loe_date = pd.Period(top_n.iloc[i].name[6], freq='M')
            if _debug: print('..with loe'.ljust(pad), loe_date)

            # make the index, from the top_n input dataframe, adding n_pers
            ind_start = top_n.columns[0]
            if _debug: print('ind start'.ljust(pad), ind_start)
            ind = pd.PeriodIndex(start=ind_start, periods = len(top_n.columns) + n_pers).to_timestamp()
            if _debug: print('ind end'.ljust(pad), ind[-1])
            if _debug: print('total periods'.ljust(pad), len(ind))

            # snip df to length of index - THERE IS A STRAY PERIOD COMING FROM SOMEWHERE

            if _debug: print('length of dfs'.ljust(pad), len(df))
            if len(df) > len(ind):
                if _debug: print('snipping df.. ..')
                df = df.iloc[:len(ind), :]
                if _debug: print("length now".ljust(pad), len(df))

            ind_end = pd.Period(ind[-1], freq='M')
            if _debug: print('index end'.ljust(pad), ind_end)

            # and now loop through the actual columns in the dataframe for the product
            for col in df:
                if zeros_to_nans: 
                    ax[i].plot(ind, zero_to_nan(df[col]))
                else:
                    ax[i].plot(ind, df[col])

            ax[i].set_title(top_n.iloc[i].name[0] + ", loe: " + str(loe_date))
            if i%4 == 0:
                ax[i].legend(['actual', 'mov ave.', 'projected'])
            pat_exp = pd.Period(top_n.iloc[i].name[6], freq='M')
            lim_0 = max(ind_start, (pat_exp - eff_total_len)).to_timestamp()
            lim_1 = max(ind_start, (pat_exp - eff_plat_len)).to_timestamp()
            lim_2 = min(ind_end, max(ind_start, (pat_exp))).to_timestamp()
            lim_3 = None

 
            if lim_1 > ind_start.to_timestamp(): 
                ax[i].axvspan(lim_0, lim_1, facecolor='g', alpha=0.1)
                # only draw the line if in scope
                if lim_1 < ind_end.to_timestamp():
                    ax[i].axvline(x=lim_1, linestyle='--', color='gray')

            if lim_2 > ind_start.to_timestamp(): 
                ax[i].axvspan(lim_1, lim_2, facecolor='r', alpha=0.1)
                
                # only draw the line if in scope
                if lim_2 < ind_end.to_timestamp():
                    ax[i].axvline(x=lim_2, color='gray')

        fig.savefig(out_folder + r + '.png')


def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x==0 else x for x in values]