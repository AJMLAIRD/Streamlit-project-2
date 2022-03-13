import streamlit as st 
import os 
import pandas as pd 


st.write("Social sentiment stock eports")

# Determine path to sentiment trending csv files

path = './Data'

# Get reports from the directionary 
files = os.listdir(path)

#Extract the reports 
files_list = [os.path.splittext(file)[0].lower() for file in files]
st.write(files_list)

# Set numerical or alphabetical filing system 
files_alpha = set([filename[0] for filename in files_list]) 

sel_alpha = st.selectbox("Latest sentiment reports", list(files_alpha)) 
filtered_files = [file for file in files_list if file[0]==sel_alpha]
sel_filtered_file = st.selectbox("Chose report", filtered_files)
if sel_filtered_file:
    st.write()
    df = pd.read_csv(f"{path}//{sel_filtered_file}.csv")
    st.DataFram(df)

# Command: streamlit run app.py