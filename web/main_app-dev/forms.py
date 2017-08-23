from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField


class IndexForm(FlaskForm):
	pass

class ParamForm(FlaskForm):
	pass

class RuleSetForm(FlaskForm):
	pass

class FullForm(FlaskForm):
	add_ruleset = SubmitField('Add a ruleset')
	submit = SubmitField('Calculate')
	new_name = StringField('new name')
	clear_all = SubmitField('clear all')
	pass
