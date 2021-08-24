from re import search
import streamlit as st

from pyzipcode import ZipCodeDatabase



import pandas as pd
from pages import utils
import base64
from datetime import datetime
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
def app():
    df1=pd.read_csv('data.csv')
    
    sd=df1['Acct_Postcode']
    
    search =ZipCodeDatabase()
    j=0
    for i in sd:
        try:
            zipcode=search[i]
            df1.at[j,'Acct_State']=zipcode.city
            j=j+1
        except:
            df1.at[j,'Acct_State']=i
            continue
    st.dataframe(df1)
    choice=st.checkbox("Save Changes")
    csvfile=df1.to_csv()
    b64 = base64.b64encode(csvfile.encode()).decode()
    new_filename = "MappedResults_{}_.csv".format(timestr)
    st.markdown("#### Download File ###")
    href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Click Here!!</a>'
    st.markdown(href,unsafe_allow_html=True)