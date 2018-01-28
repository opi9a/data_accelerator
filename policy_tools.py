import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
import matplotlib.ticker as ticker
import projection_funcs as pf
from collections import namedtuple


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


def plot_shapes_line(policy, fig=None, ax=None, figsize=None, return_fig=False, save_path=None):
    '''Makes a simple line graph of the lifecycle shapes corresponding to the scenarios in a policy
    '''
    shapes = make_shapes(policy)

    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    for s in shapes:
        ax.plot(np.arange(len(shapes[s]))/12, shapes[s]*12)
        ax.set_title("Lifecycles")
        ax.set_xlabel('years post launch')
        ax.set_ylabel('£m, annualised')
        ax.legend(shapes.columns)

    if save_path is not None:
        fig.savefig(save_path)

    if return_fig: return(fig)

##_________________________________________________________________________##


def plot_cumspend_line(start_m, n_pers, 
                        policy=None, projs=None,  
                        fig=None, ax=None, figsize=None, return_fig=False, save_path=None):
    '''Plots a  line graph of the cumulated spend corresponding to the scenarios in a policy

    Can either generate a new plot, or add to existing axis (in which case pass ax)

    Can either generate projections and index from the policy, or use existing if passed
    '''

    # get projections from policy if not passed
    if projs is None: 
        projs = project_policy(policy, start_m, n_pers)

    ind = projs.index.to_timestamp()

    # create fig and ax, unless passed (which they will be if plotting in existing grid)
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    for p in projs:
        ax.plot(ind, projs[p].values*12) # mult 12 to give annualised 

    for t in ax.get_xticklabels():
        t.set_rotation(45)

    ax.legend(['1', '2'])
    ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()])
    ax.set_title("Cumulated spend")
    ax.legend(projs.columns)  

    if save_path is not None:
        fig.savefig(save_path)

    if return_fig: return(fig)

##_________________________________________________________________________##



def plot_diffs_ann_bar(start_m, n_pers, 
                        policy=None, projs=None, diffs=None,
                        fig=None, ax=None, figsize=None, return_fig=False, save_path=None):
    '''Makes a bar chart graph of the lifecycle shapes corresponding to the scenarios in a policy
    Can either generate a new plot, or add to existing axis (in which case pass ax)
    '''

    # get projections and diffs from policy if not passed
    if projs is None: 
        if policy is None and diffs is None: 
            print('need a policy or a set of projections or diffs')
        projs = project_policy(policy, start_m, n_pers)

    if diffs is None:
        diffs = projs.iloc[:,1:].subtract(projs.iloc[:,0], axis=0)    


    ind = projs.index.to_timestamp()

    annual_projs = projs.groupby(projs.index.year).sum()
    annual_diffs = diffs.groupby(diffs.index.year).sum()

    # set the name of the counterfactual
    counterfactual_name = projs.columns[0]

    # create fig and ax, unless passed (which they will be if plotting in existing grid)
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    num_rects = len(annual_diffs.columns)
    rect_width = 0.5
    gap = 0.45
    for i, x in enumerate(annual_diffs):
        rect = ax.bar(annual_diffs.index + ((i/num_rects)*(1-gap)), annual_diffs[x], 
                        width=rect_width/num_rects) 
    ax.set_title("Difference in annual spend vs " + counterfactual_name +", £m")
    ax.tick_params(axis='x', bottom='off')
    ax.grid(False, axis='x')
    # for t in ax.get_xticklabels():
    #     t.set_rotation(45)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()])
    ax.legend(annual_diffs.columns)
    if len(annual_diffs.columns)>2:  ax.legend(annual_diffs.columns)

    if save_path is not None:
        fig.savefig(save_path)

    if return_fig: return(fig)

##_________________________________________________________________________##


def plot_impact_grid3(policy, start_m, n_pers, projs=None, diffs=None,
                        save_path=None, plot_diffs=True, return_fig=False,
                        table=False):
    '''Plots a grid of charts.

    Going to change this to use individual plotting functions 
    for each chart commonly needed, so can then choose whatever grid layout
    '''
    
    if projs is None: projs = project_policy(policy, start_m, n_pers)
    ind = projs.index.to_timestamp()
    if diffs is None: diffs = projs.iloc[:,1:].subtract(projs.iloc[:,0], axis=0)    

# plot all shapes and cumulated projections
# for diffs, calc vs first columnb
    
    annual_projs = projs.groupby(projs.index.year).sum()
    annual_diffs = diffs.groupby(diffs.index.year).sum()


    tab_rows = 2
    if plot_diffs:
        tab_rows +=1
        if table:
            tab_rows +=1

    fig = plt.figure(figsize=(12,tab_rows*5))
    rcParams['axes.titlepad'] = 12

    ax0 = plt.subplot2grid((tab_rows,2), (0, 0))
    plot_shapes_line(policy, ax=ax0)

    ax1 = plt.subplot2grid((tab_rows,2), (0, 1))
    plot_cumspend_line(policy, start_m=start_m, n_pers=n_pers, ax=ax1)

    if plot_diffs:
        ax2 = plt.subplot2grid((tab_rows,2), (1, 0), colspan=2)
        plot_diffs_ann_bar(policy, start_m=start_m, n_pers=n_pers, ax=ax2, projs=projs, diffs=diffs)

    if table:
        tab = plt.subplot2grid((tab_rows,2), (2, 0), colspan=2)
        tab.set_frame_on(False)
        tab.set_xticks([])
        tab.set_yticks([])

        rowvals = ["{:0,.0f}".format(x) for x in annual_diffs.iloc[:,0].values]
        the_table = tab.table(cellText=[rowvals], rowLabels=['spend, £m'],
                            loc='top', bbox=[0, -1, 1, 1])
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(10)

    # fig.text(0.13,0.8,'here is text')

    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    if save_path is not None:
        fig.savefig(save_path)

    if return_fig:
        return fig

##_________________________________________________________________________##


def project_policy(policy, start_m, n_pers, diffs_out=False, annual=False):  
    '''For a list of scenarios (with a start month and number of periods),
    returns a df with projections.
    
    The function orients individual projections according to the launch delays
    across the whole set of scenarios, inserting spacers as necessary.
    
    PARAMETERS
    
    policy     : a list of scenarios (see Scenario class)
    
    start_m    : start month for the projections 
                    (i.e. period zero for the scenario with lowest launch delay)
               : format eg '3-2019' is march 2019 (pd.Period standard)
               
    n_pers     : number of periods to project.  Normally months
    
    '''
    min_delay = 0
    out = []

    for s in policy:
        # make it work with a dict or a list
        if isinstance(policy, dict): s = policy[s]
        if s.launch_delay < min_delay: min_delay = s.launch_delay
   
    for s in policy:
        if isinstance(policy, dict): s = policy[s]
        spacer = s.launch_delay - min_delay
        spaced_shape = np.concatenate([np.zeros(spacer), s.get_shape()])
        out.append(pd.Series(pf.get_forecast(spaced_shape, coh_growth=s.cohort_gr/12, 
                                                term_growth=s.terminal_gr/12, 
                                                l_start=0, l_stop=n_pers, 
                                                name=s.name)))

    ind = pd.PeriodIndex(start=start_m, freq='M', periods=n_pers)
    df = pd.DataFrame(out).T
    df.index = ind

    if diffs_out:
        df = df.iloc[:,1:].subtract(df.iloc[:,0], axis=0)

    if annual:
        df = df.groupby(df.index.year).sum()

    return df


##_________________________________________________________________________##


def make_shapes(policy):
    '''Helper function to generate arrays for plotting to visualise the shapes
    (rather than for further calculations)

    Input is a list of Scenario instances
    Returns the corresponding shapes, after aligning across launch delays
    '''
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
        if isinstance(policy, dict): s = policy[s]        
        
        # assemble the elements
        spacer = s.launch_delay - min_delay
        main_phase = make_profile_shape(s.peak_sales_pm, s.shed)

        # calculate any additional terminal growth period, so all scenarios line up
        tail = 12 + max_delay - s.launch_delay
        terminus = main_phase[-1] * ((1+s.terminal_gr)** np.arange(1,tail))
        
        ser = pd.Series(np.concatenate([np.zeros(spacer), main_phase, terminus]), name=s.name)
        # put together in a pd.Series
        out.append(ser)
    

    return pd.DataFrame(out).T

##_________________________________________________________________________##


def make_profile_shape(peak_sales_pa, shed, _debug=False):
    '''Returns a profile (np.array) from input description of lifecycle shape
    (shed namedtuple with fields 'shed_name', 'uptake_dur', 'plat_dur', 'gen_mult',
    and peak sales.
    '''
    prof = np.array([float(shed.uptake_dur)] * (shed.uptake_dur+shed.plat_dur+1))
    prof[:shed.uptake_dur-1] = np.arange(1,shed.uptake_dur)
    prof[-1] = (shed.uptake_dur * shed.gen_mult)

    return prof * peak_sales_pa / max(prof)


##_________________________________________________________________________##



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


class Scenario():
    '''Stores information required to define a policy scenario.  I

    CONSTRUCTOR ARGUMENTS

    name            : the name of the scenario 

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
    def __init__(self, name, shed, terminal_gr, launch_delay, cohort_gr, peak_sales_pa):
        self.name = name
        self.shed = shed
        self.terminal_gr = terminal_gr
        self.cohort_gr = cohort_gr
        self.peak_sales_pm = peak_sales_pa / 12
        self.launch_delay = launch_delay


    def set_var(self, set_dict):
        for v in set_dict:
            # print('setting ', v, 'to ', set_dict[v])
            if v=='name': self.name=set_dict[v]
            if v=='shed': self.shed=set_dict[v]
            if v=='terminal_gr': self.terminal_gr=set_dict[v]
            if v=='cohort_gr': self.cohort_gr=set_dict[v]
            if v=='launch_delay': self.launch_delay=set_dict[v]
            if v=='peak_sales_pa': 
                self.peak_sales_pa= set_dict[v]
                self.peak_sales_pm = self.peak_sales_pa / 12


    
    def __str__(self):
        pad1 = 20
        pad2 = 20
        return "\n".join([
            "Scenario name:".ljust(pad1) + str(self.name).rjust(pad2),
            "peak sales pa:".ljust(pad1) + str(self.peak_sales_pm*12).rjust(pad2),
            "shed:".ljust(pad1),
            "  - shed name:".ljust(pad1) + self.shed.shed_name.rjust(pad2),
            "  - uptake_dur:".ljust(pad1) + str(self.shed.uptake_dur).rjust(pad2),
            "  - plat_dur:".ljust(pad1) + str(self.shed.plat_dur).rjust(pad2),
            "  - gen_mult:".ljust(pad1) + str(self.shed.gen_mult).rjust(pad2),
            "terminal_gr:".ljust(pad1) + str(self.terminal_gr).rjust(pad2),
            "launch_delay:".ljust(pad1) + str(self.launch_delay).rjust(pad2),
            "cohort_gr:".ljust(pad1) + str(self.cohort_gr).rjust(pad2),

        ])
 

    def __repr__(self):
        return self.__str__()

    def get_shape(self):
        return make_profile_shape(self.peak_sales_pm, self.shed)


##_________________________________________________________________________##b

def cost_ratio(r, k1=None, k2=None, ratio=None):
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
    if ratio is None:
        if (k1 is None) or (k2 is None): 
            print('need a ratio or k1 and k2')
            return
        
        else: ratio = k2/k1


    return ((1-r)*(ratio)) + r