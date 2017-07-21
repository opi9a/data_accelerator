from flask.ext.wtf import Form 
from wtforms.fields import TextField, SubmitField
from wtforms.validators import Required

class ContactForm(Form):
    profile = TextField("Profile", [Required("Enter a profile")])
    submit = SubmitField("Send")
