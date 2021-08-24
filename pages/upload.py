
import streamlit as st
import numpy as np
import pandas as pd
from pages import utils
from datetime import datetime

def app():
    
    st.title("Upload Data File")
    fileuploade=st.file_uploader("Upload CSV",type=["csv","xlsx"])
    global Data
    
    
    if fileuploade is not None:
        
        file_details={"File Name":fileuploade.name,"File Type":fileuploade.type,"File Size":fileuploade.size}
        Data=pd.read_csv(fileuploade)
        
        #bert(df)
            
        source,Target=st.beta_columns(2)
        source.header("Source")
        Target.header("Target")
        col1,col2=st.beta_columns(2)
        Source=Data.columns.values.tolist()
        
        
        oSELECTED = col1.multiselect('Select',Source)
        
          

        n=pd.read_csv('TargetDataBasecsv.csv')
        target=n.columns.values.tolist()
            
        for p in target:
            col2.write(p)
        st.title("Select the column which has dates")
        col4,col5=st.beta_columns(2)
        
        Source.insert(0,'None of the above')
        datt=col4.radio("Select",Source)
        col4.success(datt)
        

    
            

            
            
    if st.button("Load Data"):
        st.success("Data Loading Successful...")
        Data.drop(columns=oSELECTED,axis=1,inplace=True)
        if datt=='None of the above':
            pass
        else:
            
            ar=Data[datt]
        
        
            for dt in ar:
                datee=dt
                form=['%d-%m-%Y','%m-%d-%Y','%B %d,%Y','%m/%d/%Y','%d/%m/%Y','%d-%m-%y','%d.%m.%Y','%m.%d.%Y','%d.%m.%y','%m.%d.%y','%m-%d-%y','%d/%m/%y','%m/%d/%y','%Y/%m/%d']
                i=0
                while int(i)<=int(len(form)): 
                    try:
                        date_object = datetime.strptime(datee,form[i])
                        g = pd.to_datetime(date_object, format='%d%m%y')
                    
                        Data[datt].replace(datee,g.date())
                        break
                    except:
        
                        i=i+1
        
        # Raw data 

        
        

        # Then, drop the column as usual.

            #Data.drop(["a"], axis=1, inplace=True)

        Data.to_csv('data.csv',index=False)
        Source.remove('None of the above')
        dd=pd.read_csv('data.csv')
        st.dataframe(dd)