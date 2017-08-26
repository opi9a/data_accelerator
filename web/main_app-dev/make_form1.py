from flask_wtf import FlaskForm
from wtforms import Form
from wtforms import StringField, SubmitField, SelectField, FormField

xForm = FlaskForm # or Form

def make_form1(rulesets, rfuncs=['','r_profile', 'r_terminal', 'r_fut']): #NB rulesets is the dict of underlying objects
                                #will need to pass actual rfuncs list (just in r_funcs.py right now)    

    # first make an empty form - this will be filled and returned
    FullForm=None
    class FullForm(xForm):
        add_ruleset = SubmitField('Add a ruleset')
        submit = SubmitField('Calculate')
        new_name = StringField('new name')
        clear_all = SubmitField('clear all')
        
    # can also make the IndexForm.  Fields are actually invariant within a df, 
    # (i.e. across rulesets applied to one df) but don't yet know if a df is 
    # available.  This way reads a new IndexForm for each ruleset.  
    # NB this DOES ALLOW DIFFERENT DATAFRAMES - depends only on the rulesets.
    class IndexForm(xForm):
        pass

    # now iterate through the rulesets, making RuleSet forms to add to the FullForm class
    for rset in rulesets:
        RuleSetForm=None # feel like would be good to re-initialise
        class RuleSetForm(xForm):
            #first the invariants (NB, that means the *fields* are invariant, not their contents)
            rname = StringField(default=rulesets[rset])
            rfunc = SelectField(choices=list(zip(rfuncs,rfuncs)))
        
        # make and add the index_slice field
        
        for i in rulesets[rset].parent_df.index.names:
            setattr(IndexForm, i, StringField(i.title()))
            # print('setting', i)
            pass
        
        setattr(RuleSetForm, 'index_slice', FormField(IndexForm))

        # set up a form for the parameters, to add to the RuleSet
        ParamForm=None
        class ParamForm(xForm):
            pass

        # check if there is a function assigned in the RuleSet object
        if rulesets[rset].func is not None:

            # if so, set up ParamForm with the function parameters (just use Strings for now)
            # NOT SURE GET_PARAMS IS RIGHT - CHECK THIS
            for p in rulesets[rset].get_params():
                setattr(ParamForm, p, StringField(p.title())) # can set up to use specific field types idc
            
        # add to the RuleSetForm (an empty form if necessary)
        setattr(RuleSetForm, 'params', FormField(ParamForm))

        # add the finished RuleSetForm to the FullForm
        setattr(FullForm, rulesets[rset].name, FormField(RuleSetForm))


    return FullForm()


    