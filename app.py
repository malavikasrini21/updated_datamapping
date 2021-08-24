import streamlit as st

# Custom imports 
from multipage import MultiPage
from pages import upload,model,Home,state # import your pages here
#from pages import *
# Create an instance of the app 
app = MultiPage()

# Title of the main page

source=[]
mod=''
# Add all your applications (pages) here
app.add_page("Home", Home.app)
app.add_page("Upload Data", upload.app)
app.add_page("Model Selector", model.app)
app.add_page("Display", state.app)
#app.add_page("Results", Results.app)


# The main app
app.run()