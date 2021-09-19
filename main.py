import streamlit as st
import pandas as pd
#import plotly.figure_factory as ff
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from  sklearn.metrics import r2_score
header=st.beta_container()
dataset=st.beta_container()
features=st.beta_container()
modeltraining =st.beta_container()
with header :
    st.title("welcome")
    st.text("this is a header")   
with dataset :
    st.header('NYC DATA SET')
    df= pd.read_excel("data/automobile.xlsx")
    st.write(df.head(20))
with features :
    st.header('FEATURES')
    st.subheader("barchart")
    x=df["highway-mpg"]
    st.bar_chart(x.head())
    ############
    #st.subheader("regression")
    #fig=sns.regplot(x="highway-mpg",y="price",data=df)
    #st.plotly_chart(fig, use_container_width=True)
    ############
    st.markdown('* ** first feature: ** this is the first feature on streamlit:')
with modeltraining :
    st.header('MODEL')
    st.text("this is the model")    
    #testing
    chart_data = pd.DataFrame(np.random.randn(20, 3),
    columns=['a', 'b', 'c'])
    st.area_chart(chart_data)
    sel_col,disp_col=st.beta_columns(2)
    max_depth=sel_col.slider("whats the max depth of the model",min_value=10,max_value=100,value=20,step=10)
    n_estimator=sel_col.selectbox('how many trees: ',options=[1,2,3,'No limits'],index=0)
    input=sel_col.text_input("what the hell")