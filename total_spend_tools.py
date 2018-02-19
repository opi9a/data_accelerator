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
                               f_args = dict(shed=biol_shed, term_gr=biol_term_gr))

    rsets['nonbiol_sec'] = RuleSet.RuleSet(df, name='nonbiol_sec', 
                               index_slice = dict(biol=False, setting='secondary', start_month=slice(None, cut_off-1, None)),
                               func = rf.r_trend,
                               f_args = dict(shed=non_biol_shed, term_gr=non_biol_term_gr))

    rsets['biol_prim'] = RuleSet.RuleSet(df, name='biol_prim', 
                               index_slice = dict(biol=True, setting='primary', start_month=slice(None, cut_off-1, None)),
                               func = rf.r_trend,
                               f_args = dict(shed=biol_shed, term_gr=biol_term_gr))

    rsets['nonbiol_prim'] = RuleSet.RuleSet(df, name='nonbiol_prim', 
                               index_slice = dict(biol=False, setting='primary', start_month=slice(None, cut_off-1, None)),
                               func = rf.r_trend,
                               f_args = dict(shed=non_biol_shed, term_gr=non_biol_term_gr))

    # future launches rulesets
    rsets['biol_sec_fut'] = RuleSet.RuleSet(df, name='biol_sec_fut', 
                               index_slice = dict(biol=True, setting='secondary', start_month=slice(cut_off, None, None)),
                               func = rf.r_fut_tr,
                               f_args = dict(shed=biol_shed, term_gr=biol_term_gr, coh_gr=biol_coh_gr/12, cut_off=cut_off))

    rsets['nonbiol_sec_fut'] = RuleSet.RuleSet(df, name='nonbiol_sec_fut', 
                               index_slice = dict(biol=False, setting='secondary', start_month=slice(cut_off, None, None)),
                               func = rf.r_fut_tr,
                               f_args = dict(shed=non_biol_shed, term_gr=non_biol_term_gr, coh_gr=non_biol_coh_gr/12, cut_off=cut_off))

    rsets['biol_prim_fut'] = RuleSet.RuleSet(df, name='biol_prim_fut', 
                               index_slice = dict(biol=True, setting='primary', start_month=slice(cut_off, None, None)),
                               func = rf.r_fut_tr,
                               f_args = dict(shed=biol_shed, term_gr=biol_term_gr, coh_gr=biol_coh_gr/12, cut_off=cut_off))

    rsets['nonbiol_prim_fut'] = RuleSet.RuleSet(df, name='nonbiol_prim_fut', 
                               index_slice = dict(biol=False, setting='primary', start_month=slice(cut_off, None, None)),
                               func = rf.r_fut_tr,
                               f_args = dict(shed=non_biol_shed, term_gr=non_biol_term_gr, coh_gr=non_biol_coh_gr/12, cut_off=cut_off))
    
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