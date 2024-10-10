import streamlit as st
from dataclasses import dataclass
import pandas as pd

#File Imports
import src.Utils.File_Handlers.file as file_handler


@dataclass
class Correlation:
    def __init__(self):
        self.correlation_threshold=0.7
    @st.cache_resource
    @staticmethod
    def Correlation(self,file_path):
        self.file_path=file_path
        self.dataframe=file_handler.read_file(file_path)
        assert(isinstance(self.dataframe ,pd.DataFrame))
