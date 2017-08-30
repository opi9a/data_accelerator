from flask import Flask, render_template, request#, redirect, session
from flask_bootstrap import Bootstrap
from flask_debugtoolbar import DebugToolbarExtension
from make_form1 import make_form1

import RuleSet
import r_funcs
from projection_funcs import variablise, slicify

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('ggplot')
import pprint as pprint
from datetime import datetime
import pickle
import os

app = Flask(__name__)
bootstrap = Bootstrap(app)
app.config['SECRET_KEY'] = 'vq98ervhq98743yh'
app.debug=True
toolbar = DebugToolbarExtension(app)

df = pd.read_pickle('c:/Users/groberta/Work/data_accelerator/spend_data_proc/dfs/main_unstacked_17AUG.pkl')
cutoff = pd.Period('3-2014', freq='M')
npers = 120

pad1 = 30

rulesets = {}
# rulesets['rs1'] = RuleSet.RuleSet(df, 'rs1')
func_table={'r_profile':r_funcs.r_profile, 
        'r_terminal':r_funcs.r_terminal, 
        'r_fut':r_funcs.r_fut, }



@app.route('/', methods=['GET', 'POST'])
def home():
    outfig="" # do I really need this ??

    print("\n**********CALLING ROUTE METHOD**********")
    
    # create form - this is automatically populated (with data) from the request.POST object
    # any new fields (eg new params/args if func is changed) should be blank
    # IS THIS NEEDED?  Given that last action in function is to do exactly this, but with updated ruleset
    # - do get errors as form variable not yet declared.  But cd set to None?.
    form = make_form1(rulesets)

    print('\nFORM AFTER CREATION from make_form(ruleset). Fields populated from request, if existing there')
    pprint.pprint(form.data)

    print('\nINITIAL RULESETS')
    pprint.pprint(rulesets)

    if form.submit():
        print('\nsubmission detected')
        print('rulesets now'.ljust(pad1), [n for n in rulesets])
    
    # parse request, and update rulesets (index slices, r_func, args)
    # could use populate_obj() here, but if eg func has changed, may not work
    # also need to process eg slice inputs, to get from strings to slices
        
        # first check for command to delete all rulesets
        if form.clear_all.data:
            print("clearing all rulesets")
            rulesets.clear()

        # now get on with updating each ruleset
        print("\nPARSING FORM AND UPDATING RULESET")
        for r in rulesets:

            print('Now in:'.ljust(pad1), r)

            # check if this ruleset to be deleted
            print("delete ruleset?",  form[r].delete_ruleset.data)
            if form[r].delete_ruleset.data:
                print('deleting:'.ljust(pad1), rulesets[r].name)
                del rulesets[r]
                print('rulesets now:'.ljust(pad1), [n for n in rulesets])
                break

            # set the function (lookup from func_table to get the right expression)
            print("setting r/func to:".ljust(pad1), func_table.get(form[r]['rfunc'].data, None))
            rulesets[r].func = func_table.get(form[r]['rfunc'].data, None)
            print('params from form1:'.ljust(pad1), form[r]['params'].data)
            print('function in ruleset:'.ljust(pad1), rulesets[r].func)
            
            # copy parameters from form to f_args dict (actually param:arg pairs) in ruleset
            # need to variablise - do this first in the form, before updating the ruleset
            # note that the form will reset to a string, as these are StringFields
            for p in form[r]['params'].data:
                print("..in parameter".ljust(pad1), p)
                print("..current contents:".ljust(pad1), form[r]['params'][p].data)
                form[r]['params'][p].data = variablise(form[r]['params'][p].data)

            # now write to variablised param:args to the ruleset
            rulesets[r].f_args = form[r]['params'].data
            del rulesets[r].f_args['csrf_token']

            print("f_args in ruleset:".ljust(pad1), rulesets[r].f_args)

            # now set the index slices
            print("setting the slice".ljust(pad1), form[r]['index_slice'].data)
            for i in form[r]['index_slice'].data:
                print("in element ".ljust(pad1), i)
                form[r]['index_slice'][i].data = slicify(form[r]['index_slice'][i].data)

            # now write to the ruleset
            rulesets[r].index_slice = form[r]['index_slice'].data
            del rulesets[r].index_slice['csrf_token']

            # save if required
            print("checking if save")
            if form[r].save_ruleset.data:
                with open('rulesets/'+str(r)+'.pkl', 'wb') as f:
                    pickle.dump(rulesets[r], f, protocol=pickle.HIGHEST_PROTOCOL)



        print("\nCHECKING FOR NEW RULESETS")     
        if form.add_ruleset.data and form.new_name.data:
            print('adding new ruleset:'.ljust(pad1), form.new_name.data)
            print('number before:'.ljust(pad1), len(rulesets))
            rulesets[form.new_name.data] = RuleSet.RuleSet(df,form.new_name.data)
            print('number after:'.ljust(pad1), len(rulesets))


        print("\nCHECKING FOR RULESETS TO LOAD")     
        # loads them, but fields get blanked when form is remade -so need to add after
        print(form.load_ruleset.data, form.load_name.data)
        if form.load_ruleset.data and form.load_name.data:
            print('loading ruleset:'.ljust(pad1), form.load_name.data)
            print('number before:'.ljust(pad1), len(rulesets))
            print('rulesets/'+str(form.load_name.data))
            print('loading as ', form.load_name.data.split('.')[0])
            try:
                with open('rulesets/'+str(form.load_name.data), 'rb') as f:
                    rulesets[form.load_name.data.split('.')[0]] = pickle.load(f)

            except:
                print('could not open ', form.load_name.data)
            print('number after:'.ljust(pad1), len(rulesets))


        print('\nFINAL RULESETS')
        pprint.pprint(rulesets)


    print('\nFORM BEFORE PLOT')
    pprint.pprint(form.data)
    print('\nplot all flag:'.ljust(pad1), form.plot_all.data)

    if form.plot_all.data==True: 
        ('form.plot_all.data is TRUE')
        # form.plot_all.data=False
        out_dfs = []
        for r in rulesets:
            print("\nPLOTTING:".ljust(pad1), rulesets[r].name)
            rulesets[r].xtrap(npers)
            # print(rulesets[r].joined.head())
            pd.DataFrame(rulesets[r].joined).to_csv('output/df1.csv')
            print('info on ', r, ':'.ljust(pad1), rulesets[r].summed.info())
            out_dfs.append(rulesets[r].summed)
            print('length of out_dfs:'.ljust(pad1), len(out_dfs))

        try:
            print('out_df list is: '.ljust(pad1), out_dfs)
            df_concat = pd.concat(out_dfs, axis=1)
            print('concatted, with shape '.ljust(pad1), len(df_concat))
            df_concat.to_csv('output/dfconcat.csv')
            fig = df_concat.plot(kind='Area', stacked='True', legend=True).get_figure()
            ts = int((datetime.now() - datetime(1970,1,1)).total_seconds())
            outfig = str('static/outfig' + str(ts) + '.png')
            fig.savefig(outfig)
            print("SAVING FIG AS:".ljust(pad1), outfig)
        

        except:
            print("couldn't make a plot / figure")

        print('RULESET AFTER PLOTTING')
        pprint.pprint(rulesets)



    # update from rulesets, and populate with data
    # NB this will turn variables back to strings
    # turn off flags etc?
    print("\nRE-MAKING FORM FROM RULESETS")
    form = None
    form = make_form1(rulesets)

    # this seems redundant with bit above but needs to happen after form remade, which blanks fields
    if form.load_ruleset.data and form.load_name.data:
        print("trying to write form")
        f = form.load_name.data.split('.')[0]
        print(f)
        print(form.data[f])
        print(rulesets[f])
        form[f]['rname'].data = rulesets[f].name

        for i in rulesets[f].index_slice:
            print("..in parameter".ljust(pad1), i)
            print("..current contents:".ljust(pad1), form[f]['index_slice'][i].data)
            form[f]['index_slice'][i].data = rulesets[f].index_slice[i]

        for p in rulesets[f].f_args:
            print("..in parameter".ljust(pad1), p)
            print("..current contents:".ljust(pad1), form[f]['params'][p].data)
            form[f]['params'][p].data = rulesets[f].f_args[p]
        print('looking up ruleset func in func_table', 
            [k for k in func_table if func_table[k]==rulesets[f].func][0])
        form[f]['rfunc'].data = [k for k in func_table if func_table[k]==rulesets[f].func][0]
        form.load_ruleset.data = False
        form.load_name.data = ""

    
    pprint.pprint(form.data)


    form['add_ruleset'].data = False
    form['new_name'].data = ""
    for r in rulesets: # why do I need to do this?
        form[r]['rname'].data = r

    return render_template('main_template1.html', form=form, 
                                rulesets=[n for n in rulesets], outfig=outfig)

if __name__ == '__main__':
    app.run()

