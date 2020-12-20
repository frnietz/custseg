# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 09:43:37 2020

@author: uy308417
"""
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
import streamlit as st
from openpyxl import Workbook
import six
import os
import base64
import sys

sys.modules['sklearn.externals.six'] = six
st.set_page_config(layout="wide")

st.sidebar.title("Upload Your Sales History")

#@st.cache(suppress_st_warning=True)
def load_data(file):
    df = pd.read_csv(file, encoding="ISO-8859-1",dtype={'CustomerID': str,'InvoiceID': str})
    return df

uploaded_file = st.sidebar.file_uploader("Upload a file in csv format", type=("csv"))

st.title("Customer Segmenter")
st.subheader('Example csv file format to upload:')
url ='https://raw.githubusercontent.com/frnietz/custseg/main/Online_retail_sample.csv'
df_sample=pd.read_csv(url)
st.write(df_sample)
st.markdown("Customer Segmentation is the key for your targeted marketing and sales.  \n"
        "This app segments your customers into 10 clusters by only using your invoice history  \n"
        "Upload your data in given format and go.   \n"
        "If you need further insights and support, please contact us for our custome,tailored offer .  \n"
        )
    

if uploaded_file is not None:
    df = load_data(uploaded_file)
   
    #sns.heatmap(sales.corr(), annot=True)
    st.write('{:,} rows; {:,} columns'.format(df.shape[0], df.shape[1]))
    st.write('{:,} invoices don\'t have a customer id and will be removed'.format(df[df.CustomerID.isnull()].shape[0]))
    df.dropna(subset=['CustomerID'], inplace=True)
    df['Price'] = df['Quantity'] * df['UnitPrice']
    orders = df.groupby(['InvoiceNo', 'InvoiceDate', 'CustomerID']).agg({'Price': lambda x: x.sum()}).reset_index()
    orders.head()
    orders['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    NOW = orders['InvoiceDate'].max() + timedelta(days=1)
    period = 365
    orders['DaysSinceOrder'] = orders['InvoiceDate'].apply(lambda x: (NOW - x).days)
    aggr = {
    	'DaysSinceOrder': lambda x: x.min(),  # the number of days since last order (Recency)
    	'InvoiceDate': lambda x: len([d for d in x if d >= NOW - timedelta(days=period)]), # the total number of orders in the last period (Frequency)
    	}
    rfm = orders.groupby('CustomerID').agg(aggr).reset_index()
    rfm.rename(columns={'DaysSinceOrder': 'Recency', 'InvoiceDate': 'Frequency'}, inplace=True)
    rfm.head()
    rfm['Monetary'] = rfm['CustomerID'].apply(lambda x: orders[(orders['CustomerID'] == x) & \
    							   (orders['InvoiceDate'] >= NOW - timedelta(days=period))]\
    					   ['Price'].sum())
    
    quintiles = rfm[['Recency', 'Frequency', 'Monetary']].quantile([.2, .4, .6, .8]).to_dict()
    
    def r_score(x):
    	if x <= quintiles['Recency'][.2]:
    		return 5
    	elif x <= quintiles['Recency'][.4]:
    		return 4
    	elif x <= quintiles['Recency'][.6]:
    		return 3
    	elif x <= quintiles['Recency'][.8]:
    		return 2
    	else:
    		return 1
        
    def fm_score(x, c):
    	if x <= quintiles[c][.2]:
    		return 1
    	elif x <= quintiles[c][.4]:
    		return 2
    	elif x <= quintiles[c][.6]:
    		return 3
    	elif x <= quintiles[c][.8]:
    		return 4
    	else:
    		return 5
    
    rfm['R'] = rfm['Recency'].apply(lambda x: r_score(x))
    rfm['F'] = rfm['Frequency'].apply(lambda x: fm_score(x, 'Frequency'))
    rfm['M'] = rfm['Monetary'].apply(lambda x: fm_score(x, 'Monetary'))
    rfm['RFM Score'] = rfm['R'].map(str) + rfm['F'].map(str) + rfm['M'].map(str)
    
    segt_map = {
    	 r'[1-2][1-2]': 'hibernating',
    	 r'[1-2][3-4]': 'at risk',
    	 r'[1-2]5': 'can\'t loose',
    	 r'3[1-2]': 'about to sleep',
    	 r'33': 'need attention',
    	 r'[3-4][4-5]': 'loyal customers',
    	 r'41': 'promising',
    	 r'51': 'new customers',
    	 r'[4-5][2-3]': 'potential loyalists',
    	 r'5[4-5]': 'champions'
    }
    
    rfm['Segment'] = rfm['R'].map(str) + rfm['F'].map(str)
    rfm['Segment'] = rfm['Segment'].replace(segt_map, regex=True)
    
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    for i, p in enumerate(['R', 'F']):
        parameters = {'R':'Recency', 'F':'Frequency'}
        y = rfm[p].value_counts().sort_index()
        x = y.index
        ax = axes[i]
        bars = ax.bar(x, y, color='silver')
        ax.set_frame_on(False)
        ax.tick_params(left=False, labelleft=False, bottom=False)
        ax.set_title('Distribution of {}'.format(parameters[p]),
                     fontsize=14)
        for bar in bars:
            value = bar.get_height()
            if value == y.max():
                bar.set_color('firebrick')
            ax.text(bar.get_x() + bar.get_width() / 2,
                    value - 5,
                    '{}\n({}%)'.format(int(value), int(value * 100 / y.sum())),
                    ha='center',
                    va='top',
                    color='w')
    #plt.show()
    st.write(fig)
    
    fig, axes = plt.subplots(nrows=5, ncols=5,
                             sharex=False, sharey=True,
                             figsize=(10, 10))
    r_range = range(1, 6)
    f_range = range(1, 6)
    
    for r in r_range:
        for f in f_range:
            y = rfm[(rfm['R'] == r) & (rfm['F'] == f)]['M'].value_counts().sort_index()
            x = y.index
            ax = axes[r - 1, f - 1]
            bars = ax.bar(x, y, color='silver')
            if r == 5:
                if f == 3:
                    ax.set_xlabel('{}\nF'.format(f), va='top')
                else:
                    ax.set_xlabel('{}\n'.format(f), va='top')
            if f == 1:
                if r == 3:
                    ax.set_ylabel('R\n{}'.format(r))
                else:
                    ax.set_ylabel(r)
            ax.set_frame_on(False)
            ax.tick_params(left=False, labelleft=False, bottom=False)
            ax.set_xticks(x)
            ax.set_xticklabels(x, fontsize=8)
            for bar in bars:
                value = bar.get_height()
                if value == y.max():
                    bar.set_color('firebrick')
                ax.text(bar.get_x() + bar.get_width() / 2,
                        value,
                        int(value),
                        ha='center',
                        va='bottom',
                        color='k')
    fig.suptitle('Distribution of M for each F and R',
                 fontsize=14)
    plt.tight_layout()
    #plt.show()
    st.write(fig)
    
    segments_counts = rfm['Segment'].value_counts().sort_values(ascending=True)
    
    fig, ax = plt.subplots()
    bars = ax.barh(range(len(segments_counts)),
                   segments_counts,
                   color='silver')
    ax.set_frame_on(False)
    ax.tick_params(left=False,
                   bottom=False,
                   labelbottom=False)
    ax.set_yticks(range(len(segments_counts)))
    ax.set_yticklabels(segments_counts.index)
    for i, bar in enumerate(bars):
            value = bar.get_width()
            if segments_counts.index[i] in ['champions', 'loyal customers']:
                bar.set_color('firebrick')
            ax.text(value,
                    bar.get_y() + bar.get_height()/2,
                    '{:,} ({:}%)'.format(int(value),
                                       int(value*100/segments_counts.sum())),
                    va='center',
                    ha='left'
                   )
    
    st.write(fig)
    # count the number of customers in each segment
    segments_counts = rfm['Segment'].value_counts().sort_values(ascending=True)
    
    
    filtered_data=rfm[['Segment','CustomerID']]
    #segmented_customers = rfm.to_excel (r'C:\Users\uy308417\OneDrive - GSK\Desktop\SegmentedCustomer.xlsx', index = None, header=True)
    
    st.markdown("To download generated data and use advantages of our advanced services please see [ULS Offers](https://www.ulsinsights.com/pricing)  \n"
            "You can purchase the product or get an offer for our extended services  \n")
    
    #csv = rfm.to_csv(index=False)
    #b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    #href = f'<a href="data:file/csv;base64,{b64}" download="segmented.csv">Download CSV File</a>(right-click and save as &lt;some_name&gt;.csv)'
    #st.markdown(href, unsafe_allow_html=True)
    
else:
    st.write("upload your data in given format or contact us to handle")
