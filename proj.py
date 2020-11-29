import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

st.title("Internet news data with readers engagement") #bold

#you can insert markdowns
'''
Kaggle dataset to explore the popularity of an article befre being published online.
The dataset has been created using additional data from facebook Graph API regarding interactions such as comments, shares, etc.
(Kaggle)
'''

st.title("Exploratory data analysis")
'''
Below you can have a look at the raw dataset (by clicking on "Show dataframe")
'''
#import data
df = pd.read_csv('/home/saramasarone/Desktop/app_project/articles_data.csv', header = 0, index_col=0)
df = df.drop(columns = ['url','url_to_image', 'published_at', 'content','description', 'title' ])
df = df.dropna(how = 'any')
#button to show df or not
if st.checkbox('Show dataframe'):
    df
'''
Before diving in the data analysis you can start explore the different variables we have in this dataset.
They can be chosen with the menu below.'''

##### show df or not?
st.markdown("#### Plotting data")
columns = df.columns
option1 = st.selectbox(
    'Which feature would you like to plot as x-axis?', columns)
option2 = st.selectbox(
    'Which feature would you like to plot as y-axis?', columns)

fig, ax = plt.subplots()
chart = sns.scatterplot(df[option1], df[option2], hue =df['source_id'],  ax = ax)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45,horizontalalignment='right')
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
plt.title("Readers engagement")
st.pyplot(fig)


st.markdown("#### TSNE")
#prepare data for TSNE
st.sidebar.markdown("## Choose the perplexity")
X = df.iloc[:,3:7]

#remove Nan
X = X.dropna(how = 'any')

option_tsne = st.sidebar.selectbox(
    'Which value would you like to use?',
     [10, 20, 30, 40, 50, 60, 70, 80])

st.sidebar.markdown("### Choose the learning rate ")
option_lr = st.sidebar.slider("learning_rate", 10, 300)
st.sidebar.markdown("### Choose the n of iterations")
option_ni = st.sidebar.slider("N of iterations", 500,2000)
st.sidebar.markdown("### Verbosity")
option_verbose =st.sidebar.selectbox('Verbosity',[0,1,2])

tsne = TSNE(n_components=2, perplexity = option_tsne,learning_rate=option_lr, n_iter=option_ni, verbose = option_verbose)
X_new = tsne.fit_transform(X)

fig, ax = plt.subplots()
chart = sns.scatterplot(X_new[:,0], X_new[:,1],hue = df['source_id'], ax = ax, palette = 'muted')
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
plt.xlabel("TSNE1")
plt.ylabel("TSNE2")
plt.title("TSNE")
st.pyplot(fig)

#engagement reactions grouped by source id
grouped = df\
    .groupby(['source_id'])['source_name','top_article', 'engagement_reaction_count', 'engagement_comment_count','engagement_share_count', 'engagement_comment_plugin_count']\
    .agg('sum')
#grouped
st.markdown("#### Total interactions broken down by their single contributions")
st.bar_chart(grouped)
