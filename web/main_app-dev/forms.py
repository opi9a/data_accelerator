from flask_wtf import FlaskForm
from wtforms import SubmitField


class IndexForm(FlaskForm):
	pass

class ParamForm(FlaskForm):
	pass

class RuleSetForm(FlaskForm):
	pass

class FullForm(FlaskForm):
	submit = SubmitField()
	pass
