from flask import Flask, render_template, request, session, url_for#, session
from flask_bootstrap import Bootstrap
from flask_debugtoolbar import DebugToolbarExtension
from make_form1 import make_form1

import RuleSet
import r_funcs
from projection_funcs import variablise, slicify

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pprint as pprint
from datetime import datetime
import pickle
import os

app = Flask(__name__)
bootstrap = Bootstrap(app)
app.config['SECRET_KEY'] = 'XXXXX-XXXXX-XXXXX-XXXXX'
app.debug=True
# toolbar = DebugToolbarExtension(app)

# df = pd.read_pickle('c:/Users/groberta/Work/data_accelerator/spend_data_proc/dfs/main_unstacked_17AUG.pkl') # old df
df = pd.read_pickle('c:/Users/groberta/Work/data_accelerator/spend_data_proc/dfs/main_df_new_dates_13SEP2017a.pkl')
cutoff = pd.Period('3-2014', freq='M')
npers = 120

plt.style.use('default')

pad1 = 30

phx_adj = 1.65 # pharmex omits a bunch of spend - this is the ave adjustment reqd

scenario = {'name':'default', 'rulesets': {}, '_totals': None}

rulesets = {}
# rulesets['rs1'] = RuleSet.RuleSet(df, 'rs1')
func_table={'r_profile':r_funcs.r_profile, 
        'r_tprofile':r_funcs.r_tprofile, 
        'r_terminal':r_funcs.r_terminal, 
        'r_trend':r_funcs.r_trend, 
        'r_fut':r_funcs.r_fut, 
        'r_fut_tr':r_funcs.r_fut_tr,
        'r_trend_old': r_funcs.r_trend_old}
        # NB currently need to hard code these options in make_form1(), to get in the SelectField

outfigs={} 


def plot_rset(scen_name, rset, npers=npers):
    rset.xtrap(npers)
    print('xtrap returned')
    sum_data = rset.summed*phx_adj*12/1000000
    ind1 = rset.summed.index.to_timestamp()

    fig, axs = plt.subplots(1,2, figsize=(12,5))

    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].plot(ind1[:-2], sum_data[:-2])
    axs[0].set_ylim(0)
    axs[0].set_ylabel('£m pa (annualised rate)')

    # hack to test if this is future - which is a different shape

    if rset.joined.shape[1] > 1:   
        data2 = (rset.joined*phx_adj*12/1000000)
        ind2 = rset.joined.columns.to_timestamp()

    else: 
        data2 = rset.past*phx_adj*12/1000000
        ind2 = rset.past.columns.to_timestamp()

    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].plot(ind2, data2.T)
    axs[1].set_ylim(0)

    ts = int((datetime.now() - datetime(1970,1,1)).total_seconds())
    scen_dir = os.path.join('static/scenarios', scen_name) 
    fig_path = os.path.join(scen_dir, rset.name + str(ts) + '.png') 
    print('fig path is '.ljust(pad1), fig_path)
    rset.out_fig = fig_path
    for f in os.listdir(scen_dir):
        if f.startswith(rset.name) and f.endswith('.png'):
            print("removing", os.path.join(scen_dir, f))
            os.remove(os.path.join(scen_dir, f))
    fig.savefig(fig_path)
    print("SAVING FIG AS:".ljust(pad1), fig_path) 


def plot_all():
        out_dfs = []
        print( "PLOTTING ALL, SCEN NAME IS ", scenario['name'])
        scen_dir = os.path.join('static/scenarios', scenario['name']) 
        
        # go through the rulesets and collect the summed projections
        for r in scenario['rulesets']:
            # first check the ruleset has been plotted
            if scenario['rulesets'][r].out_fig == "":
                print("\Trying to plot uncalculated rset:".ljust(pad1), scenario['rulesets'][r].name)
                plot_rset(scenario['name'], scenario['rulesets'][r])
            
            outfigs[r] = scenario['rulesets'][r].out_fig
            print("plotted - set outfigs dict entry to".ljust(pad1), outfigs[r])
            
            out_dfs.append(scenario['rulesets'][r].summed*phx_adj*12/1000000000)
            print('length of out_dfs:'.ljust(pad1), len(out_dfs))

        # make a df of the collected sums, save as csv/pkl and plot it
        # try:
        # print('out_df list is: '.ljust(pad1), out_dfs)
        df_concat = pd.concat(out_dfs, axis=1)
        print('concatted, with shape '.ljust(pad1), len(df_concat))
        df_concat.to_csv(os.path.join(scen_dir, 'dfconcat.csv'))
        df_concat.to_pickle(os.path.join(scen_dir, 'dfconcat.pkl'))
        
        fig, ax = plt.subplots(figsize=(14,8))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        print('made axes')
        ax.stackplot(df_concat.index.to_timestamp(), 
            [df_concat.reset_index(drop=True).iloc[:,c] for c in range(len(df_concat.columns))],
            alpha=0.5)

        ax.set_ylim(0)
        ax.set_ylabel('£bn pa (annualised)')
        ax.legend(df_concat.columns,  bbox_to_anchor=[0.2, 0.8])

        
        ts = int((datetime.now() - datetime(1970,1,1)).total_seconds())
        
        outfig = os.path.join(scen_dir, 'total_' + str(ts) + "_" + '.png')

        print('looking for old plots to delete')
        for f in os.listdir(scen_dir):
            if f.startswith('total_') and f.endswith('.png'):
                print("removing", os.path.join(scen_dir, f))
                os.remove(os.path.join(scen_dir, f))

        # plot it
        fig.savefig(outfig)
        outfigs['total'] = outfig

        print("going to save outfigs as ", scen_dir + '/outfigs.pkl')
        with open(scen_dir + '/outfigs.pkl', 'wb') as f:
            pickle.dump(outfigs, f, protocol=pickle.HIGHEST_PROTOCOL)        

        active_rset = 'total'
        print("saving TOTAL fig as:".ljust(pad1), outfig)       

        # now generate the scenario sum
        df_concat.sum(axis=1).to_pickle(os.path.join(scen_dir, 'scen_sum.pkl'))


        print('RULESET AFTER PLOTTING')
        pprint.pprint(scenario['rulesets'])


@app.route('/', methods=['GET', 'POST'])
def home():

    '''OVERVIEW

    The initial invocation of `make_form()` generates a form based on the common *structure* of the request.POST 
    object form, and the rulesets dictionary.

    The request form is a superset of the ruleset dict, with metafields such as `add_ruleset`, plus the structures of all
    the actual rulesets (which correspond with the ruleset dict).

    The overlap (i.e. the element of the request.form corresponding to the rulesets dictionary) can differ
    in two ways between the two objects.  These differences determine how the form and rulesets dict are processed.

       Field data (form ahead of rulesets dict, so update ruleset dict).  
           - the request.form may have additional info from user, eg updated coefficients for fields   
           - these fields are already consistent in the common structure
           - so all that needs to happen is for the rulesets dict to be updated with the new values
           - the form generated by the initial `make_form()` will automatically be populated with this data

       Structural information (form has info for changing ruleset dict, but then form structure must update)
           - the request.form may have info requiring changes to the structure of rulesets dict (and hence the form)
               - eg add, load or delete rulesets
               - eg selection of aa new r_func for a ruleset
           - this requires more processing than just updating fields:
               - make the changes to the ruleset dict, eg make a new instance, or delete an instance
               - generate a new form, based on the updated structure
               - [ensure new form is populated with data - don't think this is currently relevant but may be in future]  

    So the process required is:

        1. Initial call to `make_form()`
            - generates form based on common structure, with user updates to fields plus structural/meta info
            - form has same structure as ruleset (for common elements, i.e. rulesets)
            - form may have different field data (incl r_funcs)
            - form may have meta instructions for changing structure (load, delete, new, plot)
            - form may have process instructions (save)

        2. Delete any rulesets marked for deletion
            - (or clear_all)
            - form and rulesets dict differ by deleted ruleset

        3. Of remaining rulesets, update data according to form fields (NB including r_func field)
            - form and rulesets dict now have consistent data where structure overlaps
            - still differ by deleted ruleset
            [currently do some pre processing in the form - variablise() and slicify() - before updating rulesets]
        
        4. Create or load any new rulesets
            - form and rulesets dict differ by these additional rulesets, plus their field data
    
        5. Call `make_form()` again, to make a new form based on the new ruleset dict structure
            - including new r_func, and new rulesets (or deleted rulesets)   

        [PLOT - this will change the rulesets, but the results don't have to come back to the form.. ..YET]
        [ insert between 2nd make_form and update form data, to enable this]
        [ iterate thru ruleset dict, calling xtrap()]

        6. Update form data
            - needs to reflect field data in loaded datasets (defaults in new?)
            - turn off any flags? (or could do after their executed)

        7. Render form to page

        CODE DOES NOT CURRENTLY FOLLOW THIS ORDER - for plotting only, I think

    '''

    active_rset = None

    print("\n***************************CALLING APP *****************************")

    print('\nINITIAL SCENARIO')
    pprint.pprint(scenario)
    
    # 1. Initial call to `make_form()`
    form = make_form1(scenario['rulesets'])

    print('\nFORM AFTER INITIAL MAKE_FORM')
    pprint.pprint(form.data)


    if form.submit(): # I think it always is
        print('\nSubmission detected')
        active_rset = session.get('last_active_rset', None)
        print('active rset'.ljust(pad1), active_rset)
        
        # check for clear all
        if form.clear_all.data:
            print("3. Clearing all rulesets")
            scenario['rulesets'].clear()
            scenario['name'] = 'default'
            scenario['_totals'] = None
            outfigs.clear()
            active_rset = None

        # iterate through the rulesets, deleting and updating
        print("\nPARSING FORM AND UPDATING RULESET")
        for r in scenario['rulesets']:

            print('Now in:'.ljust(pad1), r)

            # 2. Delete any rulesets marked for deletion
            if form[r].delete_ruleset.data:
                print('deleting:'.ljust(pad1), scenario['rulesets'][r].name)
                del scenario['rulesets'][r]
                active_rset = None
                print('rulesets now:'.ljust(pad1), [n for n in scenario['rulesets']])
                break

            # 3. Of remaining rulesets, update data according to form fields (NB including r_func field)
            #   - set the function (lookup from func_table to get the right expression)
            print("setting r/func to:".ljust(pad1), func_table.get(form[r]['rfunc'].data, None))
            scenario['rulesets'][r].func = func_table.get(form[r]['rfunc'].data, None)
            print('setting func_str'.ljust(pad1), form[r]['rfunc'].data)
            scenario['rulesets'][r].func_str = form[r]['rfunc'].data
            print('params from form1:'.ljust(pad1), form[r]['params'].data)
            print('function in ruleset:'.ljust(pad1), scenario['rulesets'][r].func)
            print('function string in ruleset:'.ljust(pad1), scenario['rulesets'][r].func_str)
            
            #   - copy parameters from form to f_args dict (actually param:arg pairs) in ruleset
            #       first need to variablise - do this first in the form, before updating the ruleset
            #       note that the form will reset to a string, as these are StringFields
            for p in form[r]['params'].data:
                # print("..in parameter".ljust(pad1), p)
                # print("..current contents:".ljust(pad1), form[r]['params'][p].data)
                form[r]['params'][p].data = variablise(form[r]['params'][p].data)

            #    - now write to variablised param:args to the ruleset
            scenario['rulesets'][r].f_args = form[r]['params'].data
            del scenario['rulesets'][r].f_args['csrf_token']

            print("f_args in ruleset:".ljust(pad1), scenario['rulesets'][r].f_args)

            #     - now set the index slice strings.  

            scenario['rulesets'][r].string_slice = form[r]['string_slice'].data
            del scenario['rulesets'][r].string_slice['csrf_token']
            scenario['rulesets'][r].slicify_string()

            # calculate and plot if reqd
            print("checking if PLOT")
            if form[r].plot_ruleset.data:
                plot_rset(scenario['name'], scenario['rulesets'][r])                
                outfigs[r] = scenario['rulesets'][r].out_fig
                print("set outfigs dict entry to".ljust(pad1), outfigs[r])
                form[r].plot_ruleset.data = False 
                active_rset = scenario['rulesets'][r].name

            # save if required
            print("checking if SAVE RULESET")
            if form[r].save_ruleset.data:
                with open('rulesets/'+str(r)+'.pkl', 'wb') as f:
                    pickle.dump(scenario['rulesets'][r], f, protocol=pickle.HIGHEST_PROTOCOL)
                form[r].save_ruleset.data == False
                active_rset = scenario['rulesets'][r].name

            # dump to xls if required
            print("checking if DUMPING to excel")
            if form[r].dump_rset_to_xls.data:
                out_root = os.path.join('static/scenarios', scenario['name'], scenario['rulesets'][r].name) 
                writer = pd.ExcelWriter(out_root + ".xlsx")
                scenario['rulesets'][r].past.to_excel(writer, 'past')
                scenario['rulesets'][r].fut.to_excel(writer, 'fut')
                scenario['rulesets'][r].joined.to_excel(writer, 'joined')
                scenario['rulesets'][r].summed.to_excel(writer, 'summed')
                writer.save()
                with open("".join([out_root, ".pkl"]), 'wb') as f:
                    pickle.dump(scenario['rulesets'][r], f, protocol=pickle.HIGHEST_PROTOCOL)                
                form[r].dump_rset_to_xls.data == False
                active_rset = scenario['rulesets'][r].name


        # 4. Create or load any new rulesets

        print("\nCHECKING FOR NEW RULESETS")     
        if form.add_ruleset.data and form.new_name.data:
            print('adding new ruleset:'.ljust(pad1), form.new_name.data)
            print('number before:'.ljust(pad1), len(scenario['rulesets']))
            scenario['rulesets'][form.new_name.data] = RuleSet.RuleSet(df,form.new_name.data)
            print('number after:'.ljust(pad1), len(scenario['rulesets']))
            active_rset = form.new_name.data


        print("\nCHECKING FOR RULESETS TO LOAD")     
        # loads them, but fields get blanked when form is remade -so need to add after
        print(form.load_ruleset.data, form.load_name.data)
        if form.load_ruleset.data and form.load_name.data:
            print('loading ruleset:'.ljust(pad1), form.load_name.data)
            print('number before:'.ljust(pad1), len(scenario['rulesets']))
            print('rulesets/'+str(form.load_name.data))
            print('loading as ', form.load_name.data.split('.')[0])
            try:
                with open('rulesets/'+str(form.load_name.data), 'rb') as f:
                    scenario['rulesets'][form.load_name.data.split('.')[0]] = pickle.load(f)
                    active_rset = form.load_name.data.split('.')[0]

            except:
                print('could not open ', form.load_name.data)
            print('number after:'.ljust(pad1), len(scenario['rulesets']))


        print('\nFINAL RULESETS')
        pprint.pprint(scenario['rulesets'])

    # plot all if flagged
    if form.plot_all.data==True: 
        plot_all()
        active_rset = 'total'

    print("checking if SAVE SCENARIO")
    if form.save_scenario.data and form.save_scenario_name.data:
        scenario['name'] = form.save_scenario_name.data
        scen_dir = os.path.join('static/scenarios/',scenario['name'])
        if not os.path.exists(scen_dir):
            os.makedirs(scen_dir)

        scen_save_path = os.path.join(scen_dir, 'scenario.pkl')

        print('going to save scenario:'.ljust(pad1), form.save_scenario_name.data)
        with open(scen_save_path, 'wb') as f:
            pickle.dump(scenario, f, protocol=pickle.HIGHEST_PROTOCOL)

        # may as well save all the rulesets too
        # need to put this into a function
        if not os.path.exists(scen_dir+'/xls'):
            os.makedirs(scen_dir+'/xls')
        for r in scenario['rulesets']:
            out_root = os.path.join(scen_dir+'/xls', r)
            print('saving xls in scenario: ', out_root)
            writer = pd.ExcelWriter(out_root + ".xlsx")
            scenario['rulesets'][r].past.to_excel(writer, 'past')
            scenario['rulesets'][r].fut.to_excel(writer, 'fut')
            scenario['rulesets'][r].joined.to_excel(writer, 'joined')
            scenario['rulesets'][r].summed.to_excel(writer, 'summed')
            writer.save()

        form.save_scenario.data = False
        form.save_scenario_name.data = None


    print("checking if LOAD SCENARIO")
    if form.load_scenario.data and form.load_scenario_name.data:
        scen_load_path = os.path.join('static/scenarios', form.load_scenario_name.data, 'scenario.pkl')
        print('going to load scenario:'.ljust(pad1), scen_load_path)
        scenario['rulesets'].clear()
        outfigs.clear()
        active_rset = None

        for k in pd.read_pickle(scen_load_path)['rulesets']:
            print('k is ', k)
            scenario['rulesets'][k] = pd.read_pickle(scen_load_path)['rulesets'][k]
        
        scenario['name'] = form.load_scenario_name.data
        print("scenario after load:", scenario)
        plot_all()
        form.load_scenario.data = False
        form.load_scenario_name.data = ""
        form = make_form1(scenario['rulesets'])
        outfigs_temp = pd.read_pickle(os.path.join('static/scenarios', form.load_scenario_name.data, 'outfigs.pkl'))

        for k in outfigs_temp:
            outfigs[k] = outfigs_temp[k]
        print("outfigs dict loaded as, ", outfigs)



    # 5. Call `make_form()` again, to make a new form based on the new ruleset dict structure
    # NB this will turn variables back to strings
    # turn off flags etc?
    print("\nRE-MAKING FORM FROM RULESETS")
    form = None
    form = make_form1(scenario['rulesets'])

    # 6. Update form data
    if form.load_ruleset.data and form.load_name.data:
        print("trying to write form")
        f = form.load_name.data.split('.')[0]
        print(f)
        print(form.data[f])
        print(scenario['rulesets'][f])
        form[f]['rname'].data = scenario['rulesets'][f].name

        for i in scenario['rulesets'][f].string_slice:
            # print("..in parameter".ljust(pad1), i)
            # print("..current contents:".ljust(pad1), form[f]['string_slice'][i].data)
            form[f]['string_slice'][i].data = scenario['rulesets'][f].string_slice[i]

        for p in scenario['rulesets'][f].f_args:
            # print("..in parameter".ljust(pad1), p)
            # print("..current contents:".ljust(pad1), form[f]['params'][p].data)
            form[f]['params'][p].data = scenario['rulesets'][f].f_args[p]
        print('looking up ruleset func in func_table', 
            [k for k in func_table if func_table[k]==scenario['rulesets'][f].func][0])
        form[f]['rfunc'].data = [k for k in func_table if func_table[k]==scenario['rulesets'][f].func][0]
        form.load_ruleset.data = False
        form.load_name.data = ""

    # 6a update all form data if loading scenario

    if form.load_scenario.data and form.load_scenario_name.data:
        print('filling new scenario')
        print('empty form is', )
        pprint.pprint(form.data)

        for r in scenario['rulesets']:
            print('in scenario', r)
            form[r]['rname'].data = scenario['rulesets'][r].name
            
            for i in scenario['rulesets'][r].string_slice:
                # print("..in parameter".ljust(pad1), i)
                # print("..current contents:".ljust(pad1), form[r]['string_slice'][i].data)
                form[r]['string_slice'][i].data = scenario['rulesets'][r].string_slice[i]

            for p in scenario['rulesets'][r].f_args:
                print("..in parameter".ljust(pad1), p)
                print("..current contents:".ljust(pad1), form[r]['params'][p].data)
                form[r]['params'][p].data = scenario['rulesets'][r].f_args[p]

            print("function in ruleset", scenario['rulesets'][r].func)
            form[r]['rfunc'].data = scenario['rulesets'][r].func_str

        form.load_scenario.data = False
        form.load_scenario_name.data = ""

   
    pprint.pprint(form.data)


    form['add_ruleset'].data = False
    form['new_name'].data = ""
    for r in scenario['rulesets']: # why do I need to do this?
        form[r]['rname'].data = r
    
    print("Outfigs dict:".ljust(pad1), outfigs)
    print("RULESETS PASSED: ", [n for n in scenario['rulesets']])

    session['last_active_rset'] = active_rset

    return render_template('main_template1.html', form=form, active_rset=active_rset, scen_name=scenario['name'],
                                rulesets=[n for n in scenario['rulesets']], outfigs=outfigs)

@app.route('/test/')
def test():
    return render_template('tests.html')

@app.route('/scenarios/')
def scenarios():
    scen_list = [s for s in os.listdir('static/scenarios/') if (s !='default' and s[0] !='_')]
    proj_path = 'static/project'

    print('list of scenarios:', scen_list)
    dflist = []
    for p in scen_list:
        scen_sum_path = os.path.join('static/scenarios', p, 'scen_sum.pkl')
        print("opening ", scen_sum_path)
        try:
            scen_in = pd.read_pickle(scen_sum_path)
            if scen_in.name is None: scen_in.name = p
            print('found a sum with lenght: ', len(scen_in))
            dflist.append(scen_in)
            print('sums appended now: ', len(scen_sums))
        except:
            print('could not open ', p, ' for concatting')

    df = pd.concat(dflist, axis=1)
  
    df.to_csv(os.path.join(proj_path, 'project_data.csv'))
    df.to_pickle(os.path.join(proj_path, 'project_data.pkl'))

    plot = df[:-20].plot(figsize=(12,6)) # cutting off last two periods as haven't found bug that means some forecasts drop at end
    plot.spines['top'].set_visible(False)
    plot.spines['right'].set_visible(False)
    plot.set_ylabel('£bn pa (annualised)')

    # fig, ax = plt.subplots(figsize=(12,6))
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # for c in range(len(df.columns)):
    #     ax.plot(df.index.to_timestamp(), df.iloc[:,c])
    # ax.set_ylim(0)
    # ax.set_ylabel('£bn pa (annualised)')
   
    ts = int((datetime.now() - datetime(1970,1,1)).total_seconds())
    
    outfig = os.path.join(proj_path, 'project_' + str(ts) + "_" + '.png')

    print('looking for old plots to delete')
    for f in os.listdir(proj_path):
        if f.startswith('project_') and f.endswith('.png'):
            print("removing", os.path.join(proj_path, f))
            os.remove(os.path.join(proj_path, f))

    # plot it
    fig = plot.get_figure()
    fig.savefig(outfig)

    # list for plotting
    # get projection values
    # make graph

    return render_template('scenarios.html', project=scen_list, 
            outfig=os.path.join("../", outfig))

if __name__ == '__main__':
    app.run()

