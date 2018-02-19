import pandas as pd
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


    def plot_rset_projs(rs_dict, num_plots=12, n_pers=None, out_folder='figs/test_plot_rset/', save_fig=True):
    df_out = pd.DataFrame()
    gen_list = {}
    # iterate non future rsets
    for r in [x for x in rs_dict.keys() if not x.endswith('_fut')]:
        # get the top n by max sales
        print(r)
        top_n = rs_dict[r].past.loc[rs_dict[r].past.max(axis=1).sort_values(ascending=False).head(num_plots).index]
        
        #get into £m annualisex
        top_n *=12/10**8
           
        # get periods
        if n_pers is None: n_pers = len(rs_dict[r].fut)
            
        # get the LIST OF dfs from r_trend with an _out_type='df' flag
        df_list = rf.r_trend(top_n, n_pers=n_pers,
                                   shed = rs_dict[r].f_args['shed'],
                                   term_gr = rs_dict[r].f_args['term_gr'],
                                   loe_delay = rs_dict[r].f_args['loe_delay'],
                                   _out_type='df')
        
        eff_plat_len = rs_dict[r].f_args['shed'].plat_dur + rs_dict[r].f_args['loe_delay']
        eff_total_len = rs_dict[r].f_args['shed'].uptake_dur + rs_dict[r].f_args['shed'].plat_dur + rs_dict[r].f_args['loe_delay']
        print('eff plat length is ', eff_plat_len)
        # add to the list of lists of dfs
        gen_list[r] = df_list

        

        fig, ax = plt.subplots(num_plots, figsize=(12, num_plots * 6))
        for i, df in enumerate(df_list):
            for col in df:
                ind = pd.PeriodIndex(start=pd.Period('1-2007', freq='M'), periods=len(df)).to_timestamp()
                ax[i].plot(ind, df[col])
            ax[i].set_title(top_n.iloc[i].name[0])
            if i%4 == 0:
                ax[i].legend(['actual', 'mov ave.', 'projected'])
            pat_exp = pd.Period(top_n.iloc[i].name[6], freq='M')
            lim_0 = (pat_exp - eff_total_len).to_timestamp()
            lim_1 = (pat_exp - eff_plat_len).to_timestamp()
            lim_2 = (pat_exp).to_timestamp()
            lim_3 = None
            ax[i].axvline(x=lim_1, linestyle='--', color='gray')
            ax[i].axvline(x=lim_2, color='gray')
            ax[i].axvspan(lim_0, lim_1, facecolor='g', alpha=0.1)
            ax[i].axvspan(lim_1, lim_2, facecolor='r', alpha=0.1)
                
        fig.savefig(out_folder + r + '.png')
#     return gen_list 
plt.close("all")
plots_out = plot_rset_projs(dict(x=base_out['biol_sec']), num_plots=3, save_fig=False)
plots_out = plot_rset_projs(base_out, num_plots=40, save_fig=True)
