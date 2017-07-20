from flask import Flask, render_template
import pandas as pd
import numpy as np
# from spend_engine_funcs import get_forecast

# test_profile2 = np.array([1,2,5,10,11,12,12,1])

# x = get_forecast(test_profile2, 2001, 2005, 2001, 2010, 0,0,1)
# print(x)

app = Flask(__name__)

'''Can put back-end logic, calls to get_forecast() etc, 
in here
'''


@app.route('/')
def home():
	return render_template('my_layout.html')

if __name__ == '__main__':
	app.run(debug=True)