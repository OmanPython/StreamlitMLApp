import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



#Web details


#SIDEBAR
st.sidebar.write("This is the sidebar")


#MAIN VIEW

data = pd.read_csv("titanic.csv")

st.dataframe(data)
st.title("Hello World")

fig = plt.figure(figsize=(10,8))
sns.countplot(data=data, x="Survived")
st.pyplot(fig)


st.info("This is info")
st.warning("This is a warning")

#Chart





#Columns 
col1, col2 = st.columns(2)
col1.write("This is column 1")
col2.write("This is column 2")



