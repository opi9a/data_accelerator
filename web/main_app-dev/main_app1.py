from flask import Flask, render_template, request#, redirect, session
from flask_bootstrap import Bootstrap
from flask_debugtoolbar import DebugToolbarExtension
from make_form1 import make_form1

import RuleSet
import r_funcs
import projection_funcs

import pandas as pd

app = Flask(__name__)
bootstrap = Bootstrap(app)
app.config['SECRET_KEY'] = 'vq98ervhq98743yh'
app.debug=True
toolbar = DebugToolbarExtension(app)

df = pd.read_pickle('../../spend_data_proc/dfs/main_unstacked_17AUG.pkl')


rulesets = {}
rulesets['rs1'] = RuleSet.RuleSet(df, 'rs1')
rulesets['rs2'] = RuleSet.RuleSet(df, 'rs2')

@app.route('/', methods=['GET', 'POST'])
def home():
    print("REQUEST FORM", request.form)
    print("\nRULE SETS NAMES ", [n for n in rulesets])

    form = make_form1(rulesets)
    print('\nWORKED\n')
    print("FORM AFTER CREATION ", form.data)

    for r in rulesets:
        print("whole dict - type ", type(form[r].data), "data ",  form[r].data)

        for f in form[r].data:
                print("\n", f, "\n", form[r].data[f], end="\n")

    if request.method == 'POST':
        print("\nvalidated")
        print("\nadd ruleset data",form['add_ruleset'].data)
        print("\nadd ruleset data", form['new_name'].data)

        if form.clear_all.data:
            rulesets.clear()

        if form.add_ruleset.data and form.new_name.data:
            print('adding new ruleset ')
            rulesets[form.new_name.data] = RuleSet.RuleSet(df,form.new_name.data)
            print('NEW RULESETS', rulesets[form.new_name.data])
            print('len rulesets', len(rulesets))
            
    
            form = make_form1(rulesets)     
            form['new_name'].data = ""
            form['add_ruleset'].data = False
    
    for r in rulesets:
        form[r]['rname'].data=r
        

    return render_template('main_template1.html', form=form, 
                                rulesets=[n for n in rulesets])

if __name__ == '__main__':
    app.run()

