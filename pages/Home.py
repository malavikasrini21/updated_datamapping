import streamlit as st
import numpy as np
import pandas as pd
from pages import utils
def app():
    st.title("welcome")
    st.write('Semantic similarity is a metric defined over a set of documents or terms,' 
    'where the idea of distance between items is based on the likeness of their meaning '
    'or semantic content as opposed to lexicographical similarity.'
    'Schema matching is the technique of identifying objects which are semantically related.'
    'In other words, schema matching is a method of finding the correspondences between the concepts'
    'of different distributed, heterogeneous data sources. Schema matching is considered one of the basic'
    'operations for schema integration and data processing.'
    'Here we find similar dataset columns compared to each other with the option of selecting between two'
    'Models for convinience.')