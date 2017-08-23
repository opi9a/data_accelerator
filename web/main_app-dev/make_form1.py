from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField, FormField
from forms import *

import pandas as pd

def make_form1(rulesets, df): #NB rulesets is the dict of underlying objects

	# assumes df is correct
	# first construct the lowest level forms for indexing and args

	for i in df.index.names:
		setattr(IndexForm, i, StringField(i.title()))

	# could now build a blank RuleSetForm for each ruleset, then change after?
	# issue is that it's the actual form structure that needs to vary 
	# - that is, function pull down choices, and parameters

	# so may need a loop to create each RuleSetForm one by one?
	# need to be able to identify different RuleSetForms - this is new

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


	for r in rulesets: # it's a dict of RuleSet objects

		# now populate them
		# name =  name
		# df slice from ruleset.index_slice
		# function from function
		# parameters from params
		# easy! 


	# parameters depend on the function


	for r in rulesets:
		form[r].rname.data = r

	return form