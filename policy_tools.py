import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
import matplotlib.ticker as ticker
import projection_funcs as pf


def impute_peak_sales(df_slice, cutoff, shed, coh_growth_pa, term_growth_pa, ave_interval=3, 
                        _debug=False):
    '''For an input df slice (NB not by launch date), and lifecycle shape, 
    returns the impled peak sales associated with the lifecycle shape.
    
    Returns peak sales per period for one period's worth of launches.
    
    So if period is month, multiply *12*12 for peak sales pa of a year of launches.
    
    ave_interval is the interval over which terminal sales in the observed set is calculated
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
    if _debug: print('end_sales', end_sales)

    # make unscaled simulated cumulation
    shape = make_profile_shape(1, shed)
    if _debug: print("shape\n", shape)

    sim = pf.get_forecast(shape, l_start=0, l_stop=n_pers, 
                          coh_growth=coh_growth_pa/12, term_growth=term_growth_pa/12)

    if _debug: print('sim tail\n', sim.tail())

    end_sim = sim[-ave_interval:].mean()

    if _debug: print('end_sim tail', end_sim)

    return end_sales / end_sim

##_________________________________________________________________________##

def plot_impact(policy, start_m, n_pers):
    '''Plots a grid of charts.

    Going to change this to use individual plotting functions 
    for each chart commonly needed, so can then choose whatever grid layout
    '''
    
    shapes = make_shapes(policy)
    projs = project_policy(policy, start_m, n_pers)
    ind = projs.index.to_timestamp()
    diffs = projs.iloc[:,1:].subtract(projs.iloc[:,0], axis=0)    

# plot all shapes and cumulated projections
# for diffs, calc vs first column
    
    annual_projs = projs.groupby(projs.index.year).sum()
    annual_diffs = diffs.groupby(diffs.index.year).sum()
    
    fig = plt.figure(figsize=(15,9))
    rcParams['axes.titlepad'] = 12

    ax1 = plt.subplot2grid((2,2), (0, 0))
    for s in shapes:
        ax1.plot(np.arange(len(shapes[s]))/12, shapes[s]*12) # mult 12 to give annualised 
    ax1.set_title("Lifecycles")
    ax1.set_xlabel('years post launch')
    ax1.set_ylabel('£m, annualised')
    ax1.legend(shapes.columns)
    
    ax2 = plt.subplot2grid((2,2), (0, 1))
    for p in projs:
        ax2.plot(ind, projs[p].values*12) # mult 12 to give annualised 
    for t in ax2.get_xticklabels():
        t.set_rotation(45)
    ax2.legend(['1', '2'])
    # ax2.set_ylabel('£bn, annualised')
    ax2.set_yticklabels(['{:,}'.format(int(x)) for x in ax2.get_yticks().tolist()])
    ax2.set_title("Cumulated spend")
    ax2.legend(projs.columns)


    counterfactual = projs.columns[0]

    ax3 = plt.subplot2grid((2,2), (1, 0), colspan=2)
    num_rects = len(annual_diffs.columns)
    rect_width = 0.8
    gap = 0.3
    for i, x in enumerate(annual_diffs):
        rect = ax3.bar(annual_diffs.index + ((i/num_rects)*(1-gap)), annual_diffs[x], 
                        width=rect_width/num_rects) 
    ax3.set_title("Difference in annual spend vs " + counterfactual +", £m")
    ax3.tick_params(axis='x', bottom='off')
    ax3.grid(False, axis='x')
    for t in ax3.get_xticklabels():
        t.set_rotation(45)
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax3.set_yticklabels(['{:,}'.format(int(x)) for x in ax3.get_yticks().tolist()])
    ax3.legend(annual_diffs.columns)

    # fig.text(0.13,0.8,'here is text')

    fig.subplots_adjust(hspace=0.4, wspace=0.3)

##_________________________________________________________________________##


def project_policy(policy, start_m, n_pers):  
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
    for s in policy:
        if s.launch_delay < min_delay: min_delay = s.launch_delay
   
    out = []
    for s in policy:
        spacer = s.launch_delay - min_delay
        spaced_shape = np.concatenate([np.zeros(spacer), s.get_shape()])
        out.append(pd.Series(pf.get_forecast(spaced_shape, 
                                             coh_growth=s.cohort_gr/12, term_growth=s.terminal_gr/12, 
                                             l_start=0, l_stop=n_pers, 
                                             name=s.name)))
    ind = pd.PeriodIndex(start=start_m, freq='M', periods=n_pers)
    df = pd.DataFrame(out).T
    df.index = ind
    return df


##_________________________________________________________________________##


def make_shapes(policy):
    '''Helper function to generate arrays for plotting to visualise the shapes
    (rather than for further calculations)

    Input is a list of Scenario instances
    Returns the corresponding shapes, after aligning across launch delays
    '''
    min_delay = 0
    for s in policy:
        if s.launch_delay < min_delay: min_delay = s.launch_delay

    out = []
    for s in policy:
        spacer = s.launch_delay - min_delay
        out.append(pd.Series(np.concatenate([np.zeros(spacer),
            make_profile_shape(s.peak_sales_pm, s.shed)]), name=s.name))
    return pd.DataFrame(out).T

##_________________________________________________________________________##


def make_profile_shape(peak_sales_pa, shed, _debug=False):
    '''Makes a profile from input description of lifecycle shape.

    Basically builds the linear shed.

    Input is a dictionary with fields:  'uptake_dur', 'plat_dur', 'gen_mult' as normal
    '''
    prof = np.array([float(shed['uptake_dur'])] * (shed['uptake_dur']+shed['plat_dur']+1))
    prof[:shed['uptake_dur']-1] = np.arange(1,shed['uptake_dur'])
    prof[-1] = (shed['uptake_dur'] * shed['gen_mult'])

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



class Scenario():
    '''Stores information required to define a policy scenario.  I

    CONSTRUCTOR ARGUMENTS

    name            : the name of the scenario 

    shed            : dict describing the lifecycle profile
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
        
        self.shed = dict()
        self.shed['shed_name'] = shed.get('name', 'no name')
        self.shed['uptake_dur'] = shed.get('uptake_dur', None)
        self.shed['plat_dur'] = shed.get('plat_dur', None)
        self.shed['gen_mult'] = shed.get('gen_mult', None)
        
        self.terminal_gr = terminal_gr
        self.cohort_gr = cohort_gr
        self.peak_sales_pm = peak_sales_pa / 12
        self.launch_delay = launch_delay
        
    def __str__(self):
        pad1 = 20
        pad2 = 20
        return "\n".join([
            "Scenario name:".ljust(pad1) + self.name.rjust(pad2),
            "peak sales pa:".ljust(pad1) + str(self.peak_sales_pm*12).rjust(pad2),
            "shed:".ljust(pad1),
            "  - shed name:".ljust(pad1) + self.shed['shed_name'].rjust(pad2),
            "  - uptake_dur:".ljust(pad1) + str(self.shed['uptake_dur']).rjust(pad2),
            "  - plat_dur:".ljust(pad1) + str(self.shed['plat_dur']).rjust(pad2),
            "  - gen_mult:".ljust(pad1) + str(self.shed['gen_mult']).rjust(pad2),
            "terminal_gr:".ljust(pad1) + str(self.terminal_gr).rjust(pad2),
            "launch_delay:".ljust(pad1) + str(self.launch_delay).rjust(pad2),
            "cohort_gr:".ljust(pad1) + str(self.cohort_gr).rjust(pad2),

        ])
 
    def __repr__(self):
        return self.__str__()

    def get_shape(self):
        return make_profile_shape(self.peak_sales_pm, self.shed)
