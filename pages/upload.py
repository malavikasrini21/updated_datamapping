import streamlit as st
import numpy as np
import pandas as pd
from pages import utils
from datetime import datetime


def app():
    st.title("Upload Data File")
    fileuploade = st.file_uploader("Upload CSV", type=["csv", "xlsx"])
    global Data

    if fileuploade is not None:

        file_details = {"File Name": fileuploade.name, "File Type": fileuploade.type, "File Size": fileuploade.size}
        Data = pd.read_csv(fileuploade)
        st.dataframe(Data)

        source, Target = st.beta_columns(2)
        source.header("Source")
        Target.header("Target")
        col1, col2 = st.beta_columns(2)
        Source = Data.columns.values.tolist()

        oSELECTED = col1.multiselect('Select values not relevent to Target', Source)

        n = pd.read_csv('TargetDataBasecsv.csv')
        target = n.columns.values.tolist()

        for p in target:
            col2.write(p)
        
        col4, col5 = st.beta_columns(2)
        col4.title("Select the column which has dates")
        Source.insert(0, 'None of the above')
        datt = col4.radio("Select", Source)
        col4.success(datt)
        dataopt=["Month dd,YYYY","dd.mm.YYYY","dd/mm/YYYY","dd-mm-YYYY"]
        datedict={'Month dd,YYYY':'%B %d,%Y','dd.mm.YYYY':'%d.%m.%Y','dd/mm/YYYY':'%d/%m/%y','dd-mm-YYYY':'%d-%m-%Y'}
        dataopt.insert(0, 'None of the above')
        col5.title("Date format choice")
        dateoption=col5.radio("Select",dataopt)
        st.write(dateoption)
        
    if st.button("Load Data"):
        st.success("Data Loading Successful...")
        
        if datt == 'None of the above':
            pass
        else:

            ar = Data[datt]

            j = 0
            for dt in ar:
                datee = dt
                form = ['%d-%m-%Y','%M-%D-%Y','%m-%d-%Y', '%B %d,%Y', '%m/%d/%Y', '%d/%m/%Y', '%d-%m-%y', '%d.%m.%Y', '%m.%d.%Y',
                        '%d.%m.%y', '%m.%d.%y', '%m-%d-%y', '%d/%m/%y', '%m/%d/%y', '%Y/%m/%d']
                i = 0
                while int(i) <= int(len(form)):
                    try:
                        
                        date_object=datetime.strptime(datee,form[i])
                        new_format=date_object.strftime(datedict[dateoption])
                        #g = pd.to_datetime(date_object,datedict[dateoption])

                        Data.at[j, datt] = new_format
                        # Data[datt].replace(ar,newg)

                        break
                    except:

                        i = i + 1
                        continue
                j = j + 1

        # Raw data

        Data.drop(columns=oSELECTED, axis=1, inplace=True)
        st.dataframe(Data)
        Data.to_csv('data.csv', index=False)
        Source.remove('None of the above')
        dataopt.remove('None of the above')

        # Data.to_csv('pages/data.csv', index=False)


