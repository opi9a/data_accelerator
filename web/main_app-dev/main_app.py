from flask import Flask, render_template, request#, redirect, session
from flask_bootstrap import Bootstrap
from flask_debugtoolbar import DebugToolbarExtension
from make_form import make_form

import pandas as pd

app = Flask(__name__)
bootstrap = Bootstrap(app)
app.config['SECRET_KEY'] = 'vq98ervhq98743yh'
app.debug=True
toolbar = DebugToolbarExtension(app)

df = pd.read_pickle('../../spend_data_proc/dfs/main_unstacked_17AUG.pkl')
# print(df.head())

rulesets = []
df_index = df.index.names
index_dims = df_index #['a', 'b']

params = ['profile', 'gen_mult']

print(index_dims)

@app.route('/', methods=['GET', 'POST'])
def home():

	form = make_form(rulesets, index_dims, params)
	print("Rulesets are ", rulesets)
	print("index_dims are ", index_dims)
	print("df_names are ", df_index)

	if request.method == 'POST':
		print("\nvalidated")
		print("\n",form['add_ruleset'].data)
		print(form['new_name'].data)
		
		if form.add_ruleset.data and form.new_name.data:
			rulesets.append(form.new_name.data)
			form = make_form(rulesets, index_dims, params)
			form.new_name.data = ""
			form.add_ruleset.data = False

		if form.clear_all.data:
			del rulesets[:]



	else: print('\nNOT VALIDATED')

	return render_template('main_template.html', form=form, 
							rulesets=rulesets, index_dims=index_dims, params=params)

if __name__ == '__main__':
	app.run()

