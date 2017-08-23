from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField, FormField
from forms import *

def make_form(rulesets, index_dims, params):

	# first construct the lowest level forms for indexing and args
	for i in index_dims:
		setattr(IndexForm, i, StringField(i.title()))

	for p in params:
		setattr(ParamForm, p, StringField(p.title()))

	# now the next level: a single ruleset
	class RuleSetForm(FlaskForm):
		rname = StringField()
		index_slice = FormField(IndexForm)
		func = SelectField('func', choices=[('r_prof','profiler1') , 
					('r_term', 'terminal growth'), ('r_fut', 'future launches')]) 	
		args = FormField(ParamForm) 
	
	# finally assemble form fields for each ruleset to give the full form
	for r in rulesets:
		# print(r)
		setattr(FullForm, r, FormField(RuleSetForm, r.title()))

	form = FullForm()

	for r in rulesets:
		form[r].rname.data = r

	return form