from flask import Flask, render_template
from flask_wtf import Form
from wtforms import StringField, SubmitField
from wtforms.validators import Required

import pandas as pd
import numpy as np
from spend_engine_funcs import get_forecast

class MyForm(Form):
	name = StringField('And you are?')
	submit = SubmitField('Submit')

# first functionality:
# 	- input form, takes profile (csv file path) plus other data reqd
#	- carries out calculation, generates pd.Series
# 	- plots with matplotlib and saves .img
#	- serves a page with the .img (next to input form)

# first just input directly
test_profile2 = np.array([1,2,5,10,11,12,12,1])

x = get_forecast(test_profile2, 2001, 2005, 0,0,1, 2001, 2010)


app = Flask(__name__)

'''Can put back-end logic, calls to get_forecast() etc, 
in here
'''

test_input = "The sum of the profile is  " + str(x)  # passed into template

# observe how this is picked up in template html file, at {{ test_input }}

@app.route('/')
def home():
	content="this is the home page"
	return render_template('test_block.html', 
							content=content,
							prof = test_profile2, 
							out_ser=x,
							out_sum=str(sum(x)))


@app.route('/sim/', methods=['GET', 'POST'])
def sim():
	content="this is the simulator"
	dbg="XXX"

	# form = MyForm
	# name = None
	# # if form.validate_on_submit():
	# name = form.name.data
	# form.name.date = ''

	return render_template('sim_block.html', 
							# name=name, form=form,
							content=content, dbg=dbg)

@app.route('/explorer/')
def explore():
	dbg=None
	content="this is the explorer"
	return render_template('test_block.html', content=content, dbg=dbg)

@app.route('/profiler/')
def profile():
	content="this is the profiler"
	dbg="XXX"
	return render_template('test_block.html', content=content, dbg=dbg)

if __name__ == '__main__':
	app.run(debug=True)