# Customer Clustering for Profitability Prediction

This project focuses on predicting customer profitability through clustering techniques using the KMeans algorithm. The clustering is performed on customer data based on their income and score attributes.

## Project Overview

The goal of this project is to segment customers into distinct groups based on their income and score attributes. By identifying these clusters, businesses can tailor their marketing strategies, product offerings, and customer service to better meet the needs of each group, ultimately increasing profitability.

## Steps Taken

1. **Data Collection**: 
   - The first step involved gathering relevant customer data, focusing specifically on income and score attributes.

2. **Feature Selection**
   -  Selecting the most relevant features (income and score) for clustering to focus on attributes that have the greatest impact on profitability.

3. **Instantiate KMeans Model)**:
   -Create an instance of the KMeans model specifying the number of clusters.

4. **Visualize Elbow Method**:
   - Use the Yellowbrick KElbowVisualizer to visually inspect the optimal number of clusters using the elbow method.
   - Visualize Clusters: Visualize the clusters using Yellowbrick's KMeansVisualizer to understand how the data points are grouped together.

5. **Model Selection**:
   - The KMeans algorithm was chosen for clustering customers based on its simplicity and effectiveness in identifying distinct groups within the data.

6. **Model Training**:
   - The KMeans model was trained on the preprocessed data to identify clusters of customers based on their income and score attributes.

7. **Model Evaluation**:
   - The trained model was evaluated using appropriate metrics to assess its performance in predicting customer profitability. This step involves comparing predicted clusters with actual customer behavior or profitability metrics.

8. **Deployment**:
   - The model was deployed using Streamlit, allowing users to interactively explore the clustering results and visualize customer segments based on income and score attributes. The deployed application can be accessed (https://kmeansclusters-project.streamlit.app/).

## Achieving Optimal Solution

-By implementing these strategies and leveraging the power of KMeans clustering, the project achieved a high level of accuracy 97% in determining customer profitability.

## Instructions

To use the deployed model, simply visit the deployment link provided above. Input the required information and the model will predict the customers distinct group.


