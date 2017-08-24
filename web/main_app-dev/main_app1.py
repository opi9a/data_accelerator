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

rs1 = RuleSet.RuleSet(df, 'rs1')
rs2 = RuleSet.RuleSet(df, 'rs2')

    # def __init__(self, parent_df, name, index_slice={}, 
    #              func=None, f_args={}, join_output=True):


rulesets = [rs1, rs2]

@app.route('/', methods=['GET', 'POST'])
def home():

	try:
		form = make_form1(rulesets, df)
		print('\nWORKED\n')
		print(form)

	except: print("DIDNT WORK")

	# if request.method == 'POST':
	# 	# print("\nvalidated")
	# 	# print("\n",form['add_ruleset'].data)
	# 	# print(form['new_name'].data)
		
	# 	if form.add_ruleset.data and form.new_name.data:
	# 		rulesets.append(form.new_name.data)
	# 		form = make_form(rulesets, index_dims, params)
	# 		form.new_name.data = ""
	# 		form.add_ruleset.data = False

	# 	if form.clear_all.data:
	# 		del rulesets[:]



	else: print('\nNOT VALIDATED')

	return render_template('main_template.html', form=form, 
							rulesets=rulesets, index_dims=df.index.names, params=None)

if __name__ == '__main__':
	app.run()

