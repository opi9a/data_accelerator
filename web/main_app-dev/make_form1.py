from flask_wtf import FlaskForm
from wtforms import Form
from wtforms import StringField, SubmitField, SelectField, FormField
# from forms import *
import pandas as pd

xForm = Form # or FlaskForm

def make_form1(rulesets, df, rfuncs=['r_profile', 'r_terminal', 'r_fut']): #NB rulesets is the dict of underlying objects
								#will need to pass actual rfuncs list (just in r_funcs.py right now)	

	# first make an empty form - this will be filled and returned
	FullForm=None
	class FullForm(xForm):
		submit = SubmitField('calculate')
		pass

	#can also make the IndexForm ready to pass to each RuleSet - it's invariant (as a field)
	class IndexForm(xForm):
		pass

	for i in df.index.names:
		setattr(IndexForm, i, StringField(i.title()))
		pass


	# now iterate through the rulesets, making RuleSet forms to add to the FullForm class
	for rset in rulesets:
		RuleSetForm=None # feel like would be good to re-initialise
		class RuleSetForm(xForm):
			#first the invariants (NB, that means the *fields* are invariant, not their contents)
			rname = StringField(default=rset.name)
			rfunc = SelectField('functions', choices=rfuncs)
		
		# add the index_slice field
		setattr(RuleSetForm, 'index_slice', FormField(IndexForm))

		# set up a form for the parameters, to add to the RuleSet
		ParamForm=None
		class ParamForm(xForm):
			pass

		# check if there is a function assigned in the RuleSet object
		if rset.func is not None:

			# if so, set up ParamForm with the function parameters (just use Strings for now)
			# NOT SURE GET_PARAMS IS RIGHT - CHECK THIS
			for p in rset.get_params():
				setattr(ParamForm, p, StringField(p.title())) # can set up to use specific field types idc
			
		# add to the RuleSetForm (an empty form if necessary)
		setattr(RuleSetForm, 'params', FormField(ParamForm))

		# add the finished RuleSetForm to the FullForm
		setattr(FullForm, rset.name, FormField(RuleSetForm))

	return FullForm 


