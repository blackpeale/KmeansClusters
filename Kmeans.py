import pandas as pd
import streamlit as st
import joblib
import seaborn as sns
import sklearn

data = pd.read_csv('Mall_Customers (3).csv')

# - Rename the features for ease of reference
data= data.rename(columns = {'Spending Score (1-100)': 'Score', 'Annual Income (k$)': 'Income', 'CustomerID': 'ID'})

#Business Problem:  Cluster customers according to Income and Spending Score
data = data[['Income','Score']]


#!pip install yellowbrick  --q

from sklearn.cluster import KMeans # ------------------------------------------- Instantiate the KMeans algorithm
from yellowbrick.cluster import KElbowVisualizer # ----------------------------- Instantiate the plotting library

# Instantiate the clustering model and visualizer
km = KMeans(random_state=42)
visualizer = KElbowVisualizer(km, k=(2,10))

# KMEANS ALGORITHM : CLUSTERING ALGORITHM

km = KMeans(n_clusters = 5,    # ............................................... number of clusters we want
            init = 'k-means++',
            n_init = 10,
            max_iter = 100,
            random_state = 42)

clusters_predict = km.fit_predict(data) # ......................................... we fit the Kmeans algorithm on the data
data['Clusters'] = clusters_predict # ............................................. we create a new column for the cluster class

# Interpretation of Clusters
sns.set(style = 'darkgrid')
sns.scatterplot(x = data['Score'], y = data['Income'], hue = data['Clusters'], palette = 'Set2')

data['ClusterInterprete'] = data.Clusters.map({2: 'LwInc_HiSpend',
                               3: 'HiInc_LowSpend',
                               4: 'LwInc_LwSpend',
                               1: 'HiInc_HiSpend',
                               0: 'MidInc_MidSpend'})


# Split the data into train and test
from sklearn.model_selection import train_test_split
x = data.drop(['Clusters','ClusterInterprete'], axis = 1)
y = data.Clusters

xtrain, xtest, ytrain, ytest = train_test_split(x,y, train_size = 0.80, random_state = 10) # SYNTHAX TO SPLIT INTO  TRAIN AND TEST FOR MACHINE LEARNING


#--------- XGBOOST CLASSIFIER MODELLING --------------

from xgboost import XGBClassifier

xg_model = XGBClassifier()
xg_model.fit(xtrain, ytrain)


#--------------------------STREAMLIT IMPLEMENTATION-----------------

st.set_page_config(page_title='Customer Segmentation Explorer:',page_icon=':adult:')
col1, col2 = st.columns([0.1,0.9])
with col1:
    st.image('cus_ima1.png', width = 100)

html_title = """<style>.title-test{font-weight:bold;padding:5px;border-radius:6px;color:#1B4F72;}
                </style><center><h1 class="title-test">Customer Clusters Predictor</h1></center>"""
with col2:
    st.markdown(html_title,unsafe_allow_html= True)


st.markdown("<h4 style = 'margin: -30px; color:#2874A6 ; text-align: center; font-family: cursive '>Built By Chiemeziem Okeke</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<h4 style = 'margin: -30px; color:#21618C; text-align: center; font-family: helvetica '>Project Description</h4>", unsafe_allow_html = True)
st.write('Utilizing the K-means clustering algorithm, businesses can segment mall customers according to their spending habits, facilitating a deeper understanding of their shopping behaviors. This segmentation allows for the development of targeted marketing strategies, enhancing profitability and fostering growth. By clustering customers with comparable characteristics and behaviors, businesses can refine their marketing approaches to suit distinct customer segments, thereby optimizing their market outreach and maximizing returns.')

st.divider()

#st.dataframe(data,use_container_width= True)
with st.expander('Data View'):
    st.dataframe(data,use_container_width = True)


st.sidebar.image('cus_ima2.png', caption ='Please enter customer details')
                

Inc = st.sidebar.number_input('Income')

Sco = st.sidebar.number_input("Score")

# st.divider()


st.scatter_chart(data, x ='ClusterInterprete', y = 'Clusters')

st.markdown("<h4 style = 'margin: -30px; color: #1B4F72; text-align: center; font-family: helvetica '>Input Variable</h4>", unsafe_allow_html = True)
inputs = pd.DataFrame()
inputs['Income'] = [Inc]
inputs['Score'] = [Sco]

st.dataframe(inputs, use_container_width= True)


pusher = st.button('Predict Customer Cluster')

# Model Prediction

if pusher:
    predicted = xg_model.predict(inputs)
    if predicted[0]== 0:
        st.success(f'Customer is a MidInc_MidSpend')
        st.image('cus_ima3.png', width = 200)

    elif predicted[0]== 1:
        st.success(f'Customer is a HiInc_HiSpend')
        st.image('cus_ima4.png', width = 200)
    
    elif predicted[0]== 2:
        st.success(f'Customer is a LwInc_HiSpend')
        st.image('cus_ima5.png', width = 200)
        
    elif predicted[0]== 3:
        st.success(f'Customer is a HiInc_LowSpend')
        st.image ('cus_ima6.png', width = 200)
    
    else: 
        st.error(f'Customer is a LwInc_LwSpend ')
        st.image('cus_ima7.png', width = 200)

