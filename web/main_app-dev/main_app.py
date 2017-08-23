from flask import Flask, render_template, request#, redirect, session
from flask_bootstrap import Bootstrap
from flask_debugtoolbar import DebugToolbarExtension
from make_form import make_form

app = Flask(__name__)
bootstrap = Bootstrap(app)
app.config['SECRET_KEY'] = 'vq98ervhq98743yh'
app.debug=True
toolbar = DebugToolbarExtension(app)

rulesets = ['biologicals', 'non_biologicals', 'old']
index_dims = ['biol', 'launch_year']
params = ['profile', 'gen_mult']



@app.route('/', methods=['GET', 'POST'])
def home():

	form = make_form(rulesets, index_dims, params)

	if form.validate_on_submit():
		print("validated")

	return render_template('main_template.html', form=form, 
							rulesets=rulesets, index_dims=index_dims, params=params)

if __name__ == '__main__':
	app.run()

