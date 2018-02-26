import pandas as pd
import numpy as np
import inspect
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
plt.style.use('bmh')

import RuleSet
import r_funcs as rf

def make_rsets(df, params_dict, 
                xtrap=False, return_all_sums = False, return_setting_sums=False, return_sum = False,
                trim=False, _debug=False):
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
                               f_args = dict(shed=biol_shed, loe_delay=loe_delay, term_gr=biol_term_gr, coh_gr=biol_coh_gr, cut_off=cut_off))

    rsets['nonbiol_sec_fut'] = RuleSet.RuleSet(df, name='nonbiol_sec_fut', 
                               index_slice = dict(biol=False, setting='secondary', start_month=slice(cut_off, None, None)),
                               func = rf.r_fut_tr,
                               f_args = dict(shed=non_biol_shed, loe_delay=loe_delay, term_gr=non_biol_term_gr, coh_gr=non_biol_coh_gr, cut_off=cut_off))

    rsets['biol_prim_fut'] = RuleSet.RuleSet(df, name='biol_prim_fut', 
                               index_slice = dict(biol=True, setting='primary', start_month=slice(cut_off, None, None)),
                               func = rf.r_fut_tr,
                               f_args = dict(shed=biol_shed, loe_delay=loe_delay, term_gr=biol_term_gr, coh_gr=biol_coh_gr, cut_off=cut_off))

    rsets['nonbiol_prim_fut'] = RuleSet.RuleSet(df, name='nonbiol_prim_fut', 
                               index_slice = dict(biol=False, setting='primary', start_month=slice(cut_off, None, None)),
                               func = rf.r_fut_tr,
                               f_args = dict(shed=non_biol_shed, loe_delay=loe_delay, term_gr=non_biol_term_gr, coh_gr=non_biol_coh_gr, cut_off=cut_off))
    
    if xtrap or return_all_sums or return_setting_sums or return_sum: 
        for r in rsets:
            if _debug: print('xtrapping rset ', r, end=" ")
            rsets[r].xtrap(n_pers)
            if _debug: print(' ..OK')

    # if any sums reqd, make the full set
    if return_all_sums or return_setting_sums or return_sum:
        if _debug: print('making all sums')
        sums = pd.concat([rsets[x].summed for x in rsets], axis=1)
        if trim:
            sums = sums.loc[slice(pd.Period(trim[0], freq='M'), pd.Period(trim[1], freq='M'),None),:]

    # if all sums reqd, just return
    if return_all_sums:
        return sums

    # if sums by setting reqd
    elif return_setting_sums:
        if _debug: print('making sums by setting')
        sums = sums.groupby(lambda x: 'sec' in x, axis=1).sum()
        sums.columns = ['Primary', 'Secondary']
        return sums

    # if a single sum reqd
    elif return_sum: 
        if _debug: print('making single sum')
        sums = sums.sum(axis=1)
        if return_setting_sums or return_all_sums:
            print('Returning single sum only - to return all sums, or by setting you need to turn off the `return_sum` flag')
        return sums


    # default returns the whole rulesets (with or without dfs depending on if xtrap was called)    
    else: return rsets


##____________________________________________________________________________________________________________##


def load_spend_dset(path='../spend_data_proc/consol_ey_dset/spend_dset_07FEB18a.pkl', phx_adj=1.6, add_start_m=True, _debug=False):
    '''Helper function to load the psned dataset

    Option to make the general adjustments to pharmex (with a passed multiplier phx_adj)

    Currently adds a start month (optionally, default true).  This is needed to make slices of the df in the rulesest, 
    but may be better done in generating the original spend dset. (TODO) 
    '''

    pad = 30

    df = pd.read_pickle(path)

    if _debug: 
        print('\noriginal sum, £m'.ljust(pad), "{:0,.3f}".format(df.sum().sum()/10**8))
        by_setting = df.groupby(level=1).sum().sum(axis=1)/10**8
        print("by_setting.index[0]".ljust(pad), "{:0,.3f}".format(by_setting[0]))
        print("by_setting.index[1]".ljust(pad), "{:0,.3f}".format(by_setting[1]))
 
    # secondary spend before adjusting
    if _debug:
        sec1 = df.groupby(level=1).sum().loc['secondary',:].sum()
        print('\napplying ratio'.ljust(pad), phx_adj, end='\n')

    df.loc[pd.IndexSlice[:,'secondary'],:] *= phx_adj
    if _debug: 
        by_setting = df.groupby(level=1).sum().sum(axis=1)/10**8
        print('\nfinal sum, £m'.ljust(pad), "{:0,.3f}".format(by_setting.sum()))
        print("by_setting.index[0]".ljust(pad), "{:0,.3f}".format(by_setting[0]))
        print("by_setting.index[1]".ljust(pad), "{:0,.3f}".format(by_setting[1]))       
        print("\nratio actually applied ".ljust(pad), "{:0,.3f}".format(df.groupby(level=1).sum().loc['secondary',:].sum() / sec1))
        print('len'.ljust(pad), len(df))

    # add a start month
    if add_start_m:
        start_m_ind = pd.PeriodIndex(df.index.get_level_values(level='adjusted_launch_date'), freq='M', name='start_month')
        df.set_index(start_m_ind, append=True, inplace=True)

    return df

##____________________________________________________________________________________________________________##


def plot_rset_projs(rs_dict_in, selection=None, agg_filename=None, num_plots=12, n_pers=120, xlims=None, 
                out_folder='figs/test_plot_rset/', save_fig=True, _debug=False):
    
    '''Helper function for plotting projections of individual products in rulesets.

    For a dict of rulesets, sorts by max sales and takes num_plots highest.

    Then for each generates a plot containing each product in the high sales set.

    Selection: pass a list of product names instead of finding products with max sales

    agg_filename: puts together in a single image file if pass a filename (no need for a .png ending)
     - most useful wwhen passing a selection
     - will then find instances of the selection elements across all the rulesets
     - so need to extend selection names to include all instances, eg a name may occur in several. 
     - Do this with ext_selection. 
     - Also use the length of this to set the number of plots, and 
     - use ext_selection to hold the annotated names (identifying rset they were found it)
     - also use an additional iterator, agg_i, which keeps incrementing across different rsets
    '''
    if _debug: print("\nIN FUNCTION:  ".ljust(20), inspect.stack()[0][3])
    if _debug: print("..called by:  ".ljust(20), inspect.stack()[1][3], end="\n\n")   

    rcParams.update({'font.size': 8})

    pad = 35

    # first exclude future rsets
    rs_dict = {x:rs_dict_in[x] for x in rs_dict_in.keys() if 'fut' not in x}

    # set up to make a single figure if aggregating
    if agg_filename is not None:
        if selection is not None:

            # tricky.  Need to know how many across all rsets
            ext_selection = []
            for r in rs_dict_in:
                ext_selection.extend([x + " - " + r for x in rs_dict_in[r].past.index.get_level_values('molecule') 
                                        for s in selection if s in x])

            if _debug: print('extended list\n', ext_selection)

            num_plots = len(ext_selection)
            if _debug: print('num_plots (selection)'.ljust(pad), num_plots)

        else: 
            num_plots = len(rs_dict * num_plots)
            if _debug: print('num_plots (rulesets)'.ljust(pad), num_plots)

        # set up a single pdf output file (if not agg, will do one anew for each ruleset)
        pdf = PdfPages(out_folder + agg_filename + '.pdf')

    # make an iterator for if aggregating in a single fig
    agg_i = 0

    # iterate non future rsets
    for r in [x for x in rs_dict.keys() if not x.endswith('_fut')]:
        # select data by max sales
        print('\nin rset'.ljust(pad), r)

        selected = None

        # use selection if passed
        if selection is not None:
            selected = rs_dict[r].past.loc[selection]

        # otherwise take top by sales
        else:
            selected = rs_dict[r].past.loc[rs_dict[r].past.max(axis=1).sort_values(ascending=False).head(num_plots).index]
        
        if _debug: print('length of selection'.ljust(pad), len(selected))
        #get into £m annualised
        selected *=12/10**8
                      
        # get the LIST OF dfs from r_trend with an _out_type='df' flag
        df_list = rf.r_trend(selected, n_pers=n_pers,
                                   shed = rs_dict[r].f_args['shed'],
                                   term_gr = rs_dict[r].f_args['term_gr'],
                                   loe_delay = rs_dict[r].f_args['loe_delay'],
                                   _out_type='df')
        

        plat_dur = rs_dict[r].f_args['shed'].plat_dur
        total_dur = rs_dict[r].f_args['shed'].uptake_dur + plat_dur
        if _debug: print('plat length is '.ljust(pad), plat_dur)
        if _debug: print('total length is '.ljust(pad), total_dur)
        
        # make pdf for this ruleset (remember in a loop here already), if not already made one for aggregate
        if agg_filename is None:
            pdf = PdfPages(out_folder + r + '.pdf')

        # now loop through the returned dataframes - one per product (each with 3 cols / lines to plot)
        for i, df in enumerate(df_list):

            # first get rid of zeros for cleaner plotting
            df = df[df!=0]

            if _debug: print('\ndf number'.ljust(pad), i)
            if _debug: print('..corresponding product name'.ljust(pad), selected.iloc[i].name[0])

            loe_date = pd.Period(selected.iloc[i].name[6], freq='M')
            if _debug: print('..with loe'.ljust(pad), loe_date)

            # make the index, from the selected input dataframe, adding n_pers
            ind_start = selected.columns[0]
            if _debug: print('ind start'.ljust(pad), ind_start)
            ind = pd.PeriodIndex(start=ind_start, periods = len(selected.columns) + n_pers).to_timestamp()
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

            # # make an axes iterator that works with case if single plot
            # ax_it = None
            # if num_plots == 1:  ax_it = ax
            # elif agg_filename:  ax_it = ax[agg_i]
            # else:               ax_it = ax[i]

            # make a figure for this df (remember, one product, 3 lines)
            fig, ax = plt.subplots(dpi=200)

            # and now loop through the actual columns in the dataframe for the product
            for col in df:
                ax.plot(ind, df[col], linewidth=1)

            plot_name = selected.iloc[i].name[0]
            if agg_filename: plot_name = ext_selection[agg_i]
            else: plot_name = selected.iloc[i].name[0]


            ax.set_title(plot_name + ", loe: " + str(loe_date))
            if i%4 == 0:
                ax.legend(['actual', 'mov ave.', 'projected'])
            pat_exp = pd.Period(selected.iloc[i].name[6], freq='M')
            lim_0 = max(ind_start, (pat_exp - total_dur)).to_timestamp()
            lim_1 = max(ind_start, (pat_exp - plat_dur)).to_timestamp()
            lim_2 = min(ind_end, max(ind_start, (pat_exp))).to_timestamp()
            lim_3 = None

 
            if lim_1 > ind_start.to_timestamp(): 
                ax.axvspan(lim_0, lim_1, facecolor='g', alpha=0.1)
                # only draw the line if in scope
                if lim_1 < ind_end.to_timestamp():
                    ax.axvline(x=lim_1, linestyle='--', linewidth=1, color='gray')

            if lim_2 > ind_start.to_timestamp(): 
                ax.axvspan(lim_1, lim_2, facecolor='r', alpha=0.1)
                
                # only draw the line if in scope
                if lim_2 < ind_end.to_timestamp():
                    ax.axvline(x=lim_2,  linewidth=2, color='gray')

            ax.set_ylim(0)

            if xlims is not None:
                ax.set_xlim(xlims)

            agg_i +=1

            # save to the pdf, and clear the plot
            pdf.savefig()
            plt.close()

        if agg_filename is None:
            pdf.close()

    if agg_filename is not None:
        pdf.close()

    if _debug: print("\nLEAVING:  ", inspect.stack()[0][3])

##_________________________________________________________________________##


def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy.
    MAKE THIS A UFUNC"""
    return [float('nan') if x==0 else x for x in values]


##_________________________________________________________________________##


def plot_hi_lo_base(hi, lo, df, scen_names, base_params=None, base_out=None, 
    trim=('1-2010', '12-2023'), colors=['darkred', 'seagreen', 'grey'], figsize=(12,6),
    outfile=None, fig=None, ax=None, return_fig=False):

    if base_params is None and base_out is None:
        print("need either base parameters or actual data")
        return

    if base_params: base_out = tst.make_rsets(df, hi, return_sum=True, trim=trim)

    hi_out = make_rsets(df, hi, return_sum=True, trim=trim)
    lo_out = make_rsets(df, lo, return_sum=True, trim=trim)

    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ind = base_out.index.to_timestamp()
    for i, s in enumerate([hi_out, lo_out, base_out]):
        ax.plot(ind, s*12/10**11, color=colors[i], alpha=0.5)
        
    ax.legend(scen_names)

    if outfile: fig.savefig(outfile)

    if return_fig: return fig

