import streamlit as st
import pandas as pd
import numpy as np

st.cache_data(persist="disk")
def read_file(file_path:str,sheet_name:int|str=1,csv_sep:str=','):
    extensions = [".xlsx",".csv"]
    assert(any(file_path.endswith(ext) for ext in extensions))
    if file_path.endswith(".xlsx"):
        dataframe=pd.read_excel(file_path,sheet_name)
    elif file_path.endswith(".csv"):
        dataframe=pd.read_csv(file_path,sep=csv_sep)
    return dataframe
    
    