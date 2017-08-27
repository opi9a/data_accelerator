from flask import Flask, render_template, request#, redirect, session
from flask_bootstrap import Bootstrap
from flask_debugtoolbar import DebugToolbarExtension
from make_form1 import make_form1

import RuleSet
import r_funcs

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pprint as pprint

app = Flask(__name__)
bootstrap = Bootstrap(app)
app.config['SECRET_KEY'] = 'vq98ervhq98743yh'
app.debug=True
toolbar = DebugToolbarExtension(app)

df = pd.read_pickle('../../spend_data_proc/dfs/main_unstacked_17AUG.pkl')
prof1 = np.genfromtxt('c:/Users/groberta/Work/data_accelerator/profiles/prof1.csv')
cutoff = pd.Period('3-2014', freq='M')
npers = 120

rulesets = {}
rulesets['rs1'] = RuleSet.RuleSet(df, 'rs1')
func_table={'r_profile':r_funcs.r_profile, 
        'r_terminal':r_funcs.r_terminal, 
        'r_fut':r_funcs.r_fut, }

@app.route('/', methods=['GET', 'POST'])
def home():
    print("\n**********CALLING ROUTE METHOD**********")
    # create form (to receive request) - fields are empty
    form = make_form1(rulesets)

    # need to fill the fields, as they will be passed back to ruleset
    form.plot_all.data = False

    print('\nFORM AFTER INSTANTIATION - should be empty')
    pprint.pprint(form.data)

    print('\nINITIAL RULESETS')
    pprint.pprint(rulesets)

    # not using this condition yet
    if form.submit:
        print('validated')
        print('rulesets now ', [n for n in rulesets])
    
    # parse request, and update rulesets (index slices, r_func, args)
        print("\nPARSING FORM AND UPDATING RULESET")
        for r in rulesets:
            #first set the function
            rulesets[r].func = func_table.get(form[r]['rfunc'].data, None)
            print('params1', form[r]['params'].data)
            
            # now copy the parameters (shd maybe change to set_params)
            rulesets[r].set_args(form[r]['params'].data)
            
            # clear any old arguments from the ruleset
            rulesets[r].f_args.clear()

            # copy the actual arguments to the ruleset
            for p in form[r]['params'].data:
                rulesets[r].f_args[p] = form[r]['params'].data[p]
            del rulesets[r].f_args['csrf_token']
            print("f args ", rulesets[r].f_args)

            # now set the index slices
            print("the slice ", form[r]['index_slice'].data)
            rulesets[r].index_slice = form[r]['index_slice'].data
            del rulesets[r].index_slice['csrf_token']


        # check for new ruleset, add if so
        if form.add_ruleset.data and form.new_name.data:
            print('adding new ruleset ')
            rulesets[form.new_name.data] = RuleSet.RuleSet(df,form.new_name.data)
            print('NEW RULESETS', rulesets[form.new_name.data])
            print('len rulesets', len(rulesets))

        print('\nRULESETS AFTER')
        pprint.pprint(rulesets)

    else: print("\nFORM NOT VALIDATED")

    # destroy form, create new one from rulesets, and populate with data
    print("\nDESTROYING FORM AND MAKING NEW ONE FROM RULESETS")
    form = None
    form = make_form1(rulesets)
    print("NEW FORM\n")
    pprint.pprint(form.data)


    form['add_ruleset'].data = False
    form['new_name'].data = ""
    for r in rulesets:
        form[r]['rname'].data = r
        # form[r]['index_slice'].data = rulesets[r].index_slice
        # form[r]['rfunc'].data = rulesets[r].func
        # form[r]['params'].data = rulesets[r].get_params()

    print('\nFORM BEFORE PLOT')
    pprint.pprint(form.data)
    print(form.plot_all.data)

    if form.plot_all.data==True: 
        ('form.plot_all.data is TRUE')
        for r in rulesets:
            print("\nPLOTTING ", rulesets[r].name)
            print(rulesets[r])
            rulesets[r].xtrap(npers)
            print(rulesets[r])
            print(rulesets[r].joined.head())
            fig = rulesets[r].joined.T.plot(legend=False).get_figure()
            fig.savefig(str(rulesets[r].name + ".pdf"))



    return render_template('main_template1.html', form=form, 
                                rulesets=[n for n in rulesets])

if __name__ == '__main__':
    app.run()

