from flask import Flask, render_template
import pandas as pd
import numpy as np
from spend_engine_funcs import get_forecast

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
	return render_template('my_layout.html', 
							prof = test_profile2, 
							out_ser=x,
							out_sum=str(sum(x)))

if __name__ == '__main__':
	app.run(debug=True)