#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import datetime as dt


# In[4]:


df=pd.read_excel('E:\\DigiCrome\\Summer Internship NextHikes\\Project5\\telcom_data.xlsx',sheet_name='Sheet1')


# In[5]:


df.iloc[0:5,0:15]


# In[5]:


df.info()


# In[6]:


df.describe().T


# **Null Value check in DataFrame**

# In[7]:


df.isnull().sum()


# **Dimension of Data Frame in Row and Column**

# In[8]:


df.shape


# **Checking For Duplicates In BearerID and MSISDN/Number Features**

# In[9]:


df['Bearer Id'].duplicated().value_counts()


# In[10]:


df['MSISDN/Number'].duplicated().value_counts()


# In[11]:


# Find the mode of 'Bearer Id'
bearer_id_mode = df['Bearer Id'].mode()[0]  # Get the first mode if multiple exist

print(f"The mode of Bearer ID is: {bearer_id_mode}")


# **Find the 2nd Most Occuring Bearer ID**

# In[12]:


# Find the value counts of 'Bearer Id'
bearer_id_counts = df['Bearer Id'].value_counts()
bearer_id_counts

# Get the second most occurring Bearer ID
if len(bearer_id_counts) >= 2:
  second_most_occurring_bearer_id = bearer_id_counts.index[1]
  print(f"The second most occurring Bearer ID is: {second_most_occurring_bearer_id}")
else:
  print("There are not enough unique Bearer IDs to find the second most occurring one.")


# **Finding the Mode of MSISDN/Number**

# In[13]:


df['MSISDN/Number'].mode()


# **Handling  null values present in dataset**

# In[14]:



#Filling missing values with the mean
# for numerical features
for column in df.select_dtypes(include=np.number):
  df[column].fillna(df[column].mean(), inplace=True)

# for categorical features
for column in df.select_dtypes(include='object'):
  df[column].fillna(df[column].mode()[0], inplace=True)


# In[15]:



# Check for null values after handling them
print(df.isnull().sum())


# **Finding Unique values in Data Set Features**

# In[16]:


df[['Bearer Id','IMEI','MSISDN/Number']].nunique()


# **Conduct a Non-Graphical Univariate Analysis by computing dispersion
# parameters for each quantitative variable and provide useful interpretation**

# In[17]:


# Compute dispersion parameters for quantitative variables
quantitative_variables = df.select_dtypes(include=np.number).columns

for variable in quantitative_variables:
  print(f"\nDispersion parameters for {variable}:")
  print(f"Range: {df[variable].max() - df[variable].min()}")
  print(f"Interquartile Range (IQR): {df[variable].quantile(0.75) - df[variable].quantile(0.25)}")
  print(f"Variance: {df[variable].var()}")
  print(f"Standard Deviation: {df[variable].std()}")


# **Interpretation**
# 
# **Range: The range provides the spread of the data from the minimum to the maximum value. A larger range indicates a wider dispersion of data.**
# 
# 
# **IQR: The interquartile range measures the spread of the middle 50% of the data. It's less affected by outliers compared to the range.**
# 
# **Variance: It measures how far a set of numbers are spread out from their average value. A larger variance implies a greater spread.**
# 
# 
# **Standard Deviation: The square root of variance, indicating the average deviation of data points from the mean. A higher standard deviation suggests more variability in the data.**
# 

# **Conduct a Graphical Univariate Analysis by identifying the most suitable plotting options for each variable and interpreting your findings.**

# In[18]:


# Univariate Analysis for Numerical Features

# Histogram for Total UL (Uplink) and DL (Downlink)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(df['Total UL (Bytes)'], kde=True)
plt.title('Distribution of Total UL (Bytes)')

plt.subplot(1, 2, 2)
sns.histplot(df['Total DL (Bytes)'], kde=True)
plt.title('Distribution of Total DL (Bytes)')
plt.show()

# Boxplot for Duration
plt.figure(figsize=(6, 6))
sns.boxplot(y=df['Dur. (ms)'])
plt.title('Distribution of Duration (ms)')
plt.show()



# Bar plot for the most frequent occurring values
for column in df.select_dtypes(include='object'):
  if df[column].nunique() <= 10: # Limit to avoid overcrowding for very large categories
    plt.figure(figsize=(8, 4))
    value_counts = df[column].value_counts()
    sns.barplot(x=value_counts.index, y=value_counts.values)
    plt.title(f'Distribution of {column}')
    plt.xticks(rotation=45)
    plt.show()

# Interpretation:
# - Observe the counts for each category of other categorical features. This helps identify the most
#   frequent categories and understand the distribution of values within those features.


# **Interpretation:**
# **Total UL/DL:  Skewed distributions suggest most users have relatively low data usage,**
# 
# **with a few users responsible for a large portion of the overall data traffic.**
# **Duration: Boxplot reveals potential outliers (long durations), indicating some calls/sessions**
# **might be unusually longer than the average.**
# 
# # Univariate Analysis for Categorical Features
# 
# **Interpretation:**
# **Handset Manufacturer: Visualize the popularity of different handset brands. It helps
#    understand which brands are used most frequently by users.**
#    

# **Finding Skewness of the DataSet**

# In[19]:


# Calculate skewness for numerical features
skewness = df.select_dtypes(include=np.number).skew()
print(skewness)


# **Post Null Value Treatment, DataSet Description**

# In[20]:


df.describe().T


# **The top 10 handsets used by the customers**

# In[21]:



top_10_handsets = df['Handset Type'].value_counts().head(10)
print(top_10_handsets)


# In[22]:


df['Handset Type'].value_counts().head(8).plot(kind='bar')
plt.xlabel('Handset Type')
plt.ylabel('Number of Handsets')
plt.title('Top 8 Handset Manufacturers')
plt.show()


# In[28]:


plt.figure(figsize=(15,10))
sns.barplot(x=top_10_handsets.index,y=top_10_handsets.values,palette='viridis')
plt.xlabel('Handset Type')
plt.ylabel('Number of Handset')
plt.title('Top 8 Handset Manufacturer')
plt.xticks(rotation=45, ha='right')  
plt.tight_layout()
plt.show()


# **top 5 handset manufacturers**

# In[31]:


top_5_handsets=df['Handset Manufacturer'].value_counts().head(5)
print(top_5_handsets)


# In[33]:


plt.figure(figsize=(10,10))
sns.barplot(x=top_5_handsets.index,y=top_5_handsets.values,palette='viridis')
plt.xlabel('Handset Manufacturer')
plt.ylabel('Number of Handsets')
plt.title('Top 5 Handset Manufacturers')
plt.xticks(rotation=45,ha='right')
plt.tight_layout()
plt.show()


# **Top 5 handsets per top 3 handset manufacturer**

# In[36]:


# Group by Handset Manufacturer and then find the top 5 handsets for each manufacturer
top_5_handsets_per_manufacturer = df.groupby('Handset Manufacturer')['Handset Type'].value_counts().groupby(level=0).head(5)

# Filter for the top 3 handset manufacturers
top_3_manufacturers = df['Handset Manufacturer'].value_counts().head(3).index.tolist()
top_5_handsets_top_3_manufacturers = top_5_handsets_per_manufacturer[top_5_handsets_per_manufacturer.index.get_level_values('Handset Manufacturer').isin(top_3_manufacturers)]

print(top_5_handsets_top_3_manufacturers)


# **Top 5 handsets per top 3 Manufacturers in bar plot**

# In[42]:


# Create a bar plot for the top 5 handsets for each of the top 3 manufacturers
fig, ax = plt.subplots(figsize=(12, 6))

# Define a color palette for the different manufacturers
colors = ['purple', 'green', 'red']

# Iterate through the top 3 manufacturers and create bars for their top 5 handsets
for i, manufacturer in enumerate(top_3_manufacturers):
  manufacturer_data = top_5_handsets_top_3_manufacturers[manufacturer]
  ax.bar(manufacturer_data.index, manufacturer_data.values, color=colors[i], label=manufacturer)

ax.set_xlabel('Handset Type')
ax.set_ylabel('Number of Handsets')
ax.set_title('Top 5 Handsets per Top 3 Manufacturers')
ax.legend()

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[ ]:


df['Handset Manufacturer'].value_counts()


# **Task 1.1- Your employer wants to have an overview of the users’ behavior on those  applications.
# ● Aggregate per user the following information in the column
#  ○ numberof xDRsessions
#  ○ Session duration
#  ○ the total download (DL) and upload (UL) data
#  ○ thetotal data volume (in Bytes) during this session for each application**

# **Numberof xDRsessions per user and sort them on descending order**

# In[42]:


# Group by 'Bearer Id' and count the number of xDR sessions for each user
xDR_sessions_per_user = df.groupby('Bearer Id')['Dur. (ms)'].count().reset_index(name='Number of xDR Sessions')

# Sort the results in descending order of the number of xDR sessions
xDR_sessions_per_user = xDR_sessions_per_user.sort_values('Number of xDR Sessions', ascending=False)

print(xDR_sessions_per_user)


# In[ ]:





# **Numberof xDRsessions per user based on MSISDN/Number and sort them on Descending order**

# In[43]:


xDR_sessions_per_user = df.groupby('MSISDN/Number')['Dur. (ms)'].count().reset_index(name='Number of xDR Sessions')

# Sort the results in descending order of the number of xDR sessions
xDR_sessions_per_user = xDR_sessions_per_user.sort_values('Number of xDR Sessions', ascending=False).head(10)

print(xDR_sessions_per_user)


# **Total Session Duration based on Bearer Id and sort them in Descending Order**

# In[44]:


# Calculate session duration per user
session_duration_per_user = df.groupby('Bearer Id')['Dur. (ms)'].sum().reset_index(name='Total Session Duration').sort_values('Total Session Duration',ascending=False)

# Print the results
print(session_duration_per_user)


# **Total Session Duration based on MSISDN/Number and sort them in Descending Order**

# In[45]:


session_duration_per_user = df.groupby('MSISDN/Number')['Dur. (ms)'].sum().reset_index(name='Total Session Duration').sort_values('Total Session Duration',ascending=False)

# Print the results
print(session_duration_per_user)


# **Thetotal download (DL) and upload (UL) data per user and sort them in descending order Based on BearerID**

# In[46]:




# Calculate total download and upload data per user
total_dl_ul_per_user = df.groupby('Bearer Id').agg({
    'Total DL (Bytes)': 'sum',
    'Total UL (Bytes)': 'sum'
}).reset_index()

# Rename columns for clarity
total_dl_ul_per_user = total_dl_ul_per_user.rename(columns={'Total DL (Bytes)': 'Total DL','Total UL (Bytes)': 'Total UL'})

# Sort the results in descending order of total download data
total_dl_ul_per_user = total_dl_ul_per_user.sort_values(['Total DL', 'Total UL'], ascending=[False, False])

print(total_dl_ul_per_user)


# **Thetotal download (DL) and upload (UL) data per user and sort them in descending order Based on MSISDN/Number**

# In[47]:




# Calculate total download and upload data per user
total_dl_ul_per_user = df.groupby('MSISDN/Number').agg({
    'Total DL (Bytes)': 'sum',
    'Total UL (Bytes)': 'sum'
}).reset_index()

# Rename columns for clarity
total_dl_ul_per_user = total_dl_ul_per_user.rename(columns={'Total DL (Bytes)': 'Total DL','Total UL (Bytes)': 'Total UL'})

# Sort the results in descending order of total download data
total_dl_ul_per_user = total_dl_ul_per_user.sort_values(['Total DL', 'Total UL'], ascending=[False, False])

print(total_dl_ul_per_user)


# **The total Data Volume (DL+UL) per user and sort them in descending order Based on BearerID**

# In[48]:


# Calculate total data volume (download + upload) for each Bearer Id
total_data_volume_per_bearer = df.groupby('Bearer Id').agg({
    'Total DL (Bytes)': 'sum',
    'Total UL (Bytes)': 'sum'
}).reset_index()

# Calculate total data volume (DL + UL)
total_data_volume_per_bearer['Total Data Volume'] = total_data_volume_per_bearer['Total DL (Bytes)'] + total_data_volume_per_bearer['Total UL (Bytes)']

# Select only Bearer Id and total data volume
total_data_volume_per_bearer = total_data_volume_per_bearer[['Bearer Id', 'Total Data Volume']].sort_values('Total Data Volume',ascending=False)

print(total_data_volume_per_bearer)


# **The total Data Volume (DL+UL) per user and sort them in descending order Based on MSISDN/Number**

# In[48]:


# Calculate total data volume (download + upload) for each Bearer Id
total_data_volume_per_bearer = df.groupby('MSISDN/Number').agg({
    'Total DL (Bytes)': 'sum',
    'Total UL (Bytes)': 'sum'
}).reset_index()

# Calculate total data volume (DL + UL)
total_data_volume_per_bearer['Total Data Volume'] = total_data_volume_per_bearer['Total DL (Bytes)'] + total_data_volume_per_bearer['Total UL (Bytes)']

# Select only Bearer Id and total data volume
total_data_volume_per_bearer = total_data_volume_per_bearer[['MSISDN/Number', 'Total Data Volume']].sort_values('Total Data Volume',ascending=False)

print(total_data_volume_per_bearer)


# **Plotting the top 5 Bearer Id and their Data Volume consumption**

# In[64]:


# Calculate total data volume (download + upload) for each Bearer Id
total_data_volume_per_bearer = df.groupby('Bearer Id').agg({
    'Total DL (Bytes)': 'sum',
    'Total UL (Bytes)': 'sum'
}).reset_index()

# Calculate total data volume (DL + UL)
total_data_volume_per_bearer['Total Data Volume'] = total_data_volume_per_bearer['Total DL (Bytes)'] + total_data_volume_per_bearer['Total UL (Bytes)']

# Sort by Total Data Volume in descending order and get the top 5
top_5_bearers = total_data_volume_per_bearer.sort_values('Total Data Volume', ascending=False).head(5).astype(str)

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(top_5_bearers['Bearer Id'], top_5_bearers['Total Data Volume'])
plt.xlabel('Bearer Id')
plt.ylabel('Total Data Volume (Bytes)')
plt.title('Top 5 Bearer Ids by Data Volume Consumption')
plt.xticks(rotation=90, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()


# **Plot the tope 5 MSISDN/Number  Id and their Data Volume consumption**

# In[51]:




# Calculate total data volume (download + upload) for each MSISDN/Number
total_data_volume_per_msisdn = df.groupby('MSISDN/Number').agg({
    'Total DL (Bytes)': 'sum',
    'Total UL (Bytes)': 'sum'
}).reset_index()

# Calculate total data volume (DL + UL)
total_data_volume_per_msisdn['Total Data Volume'] = total_data_volume_per_msisdn['Total DL (Bytes)'] + total_data_volume_per_msisdn['Total UL (Bytes)']

# Sort by Total Data Volume in descending order and get the top 5
top_5_msisdns = total_data_volume_per_msisdn.sort_values('Total Data Volume', ascending=False).head(5).astype(str)


# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(top_5_msisdns['MSISDN/Number'], top_5_msisdns['Total Data Volume'], color=['red', 'blue', 'green', 'orange', 'purple'])
plt.xlabel('MSISDN/Number')
plt.ylabel('Total Data Volume (Bytes)')
plt.title('Top 5 MSISDNs by Data Volume Consumption')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()


# **Scatter Plot between Social Media APP with its DL and UL**

# In[58]:


# Create a new DataFrame with total DL+UL data per MSISDN/Number
total_dl_ul_per_user = df.groupby('MSISDN/Number').agg({
    'Total DL (Bytes)': 'sum',
    'Total UL (Bytes)': 'sum'
}).reset_index()
total_dl_ul_per_user['Total Data Volume'] = total_dl_ul_per_user['Total DL (Bytes)'] + total_dl_ul_per_user['Total UL (Bytes)']

# Create a scatter plot for each application against total DL+UL data
application_columns = [
    'Social Media DL (Bytes)', 'Social Media UL (Bytes)', 'Youtube DL (Bytes)',
    'Youtube UL (Bytes)', 'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
    'Google DL (Bytes)', 'Google UL (Bytes)', 'Email DL (Bytes)',
    'Email UL (Bytes)', 'Gaming DL (Bytes)', 'Gaming UL (Bytes)'
]

for app_col in application_columns:
    # Aggregate application data per Bearer Id
    app_data_per_user = df.groupby('MSISDN/Number')[app_col].sum().reset_index()

    # Merge with total DL+UL data
    merged_df = pd.merge(total_dl_ul_per_user, app_data_per_user, on='MSISDN/Number')

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Total Data Volume', y=app_col, data=merged_df)
    plt.xlabel('Total DL+UL Data (Bytes)')
    plt.ylabel(app_col)
    plt.title(f'Relationship between Total Data Volume and {app_col}')
    plt.show()


# Calculate and print the correlation matrix
correlation_matrix = merged_df[['Total Data Volume', app_col]].corr()
print(f"Correlation Matrix for {app_col}:\n{correlation_matrix}\n")


# **Bivariate Analysis– explore the relationship between  Social Media,  Google, Email, YouTube, Netflix, Gaming ,other application  Upload, Download, and total bytes & the total DL+UL**
# 

# In[54]:




# Create a new DataFrame with total DL+UL data per MSISDN/Number
total_dl_ul_per_user = df.groupby('MSISDN/Number').agg({
    'Total DL (Bytes)': 'sum',
    'Total UL (Bytes)': 'sum'
}).reset_index()
total_dl_ul_per_user['Total Data Volume'] = total_dl_ul_per_user['Total DL (Bytes)'] + total_dl_ul_per_user['Total UL (Bytes)']

# Create a scatter plot for each application against total DL+UL data
application_columns = [
    'Social Media DL (Bytes)', 'Social Media UL (Bytes)', 'Youtube DL (Bytes)',
    'Youtube UL (Bytes)', 'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
    'Google DL (Bytes)', 'Google UL (Bytes)', 'Email DL (Bytes)',
    'Email UL (Bytes)', 'Gaming DL (Bytes)', 'Gaming UL (Bytes)'
]

for app_col in application_columns:
    # Aggregate application data per MSISDN/Number
    app_data_per_user = df.groupby('MSISDN/Number')[app_col].sum().reset_index()

    # Merge with total DL+UL data
    merged_df = pd.merge(total_dl_ul_per_user, app_data_per_user, on='MSISDN/Number')

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Total Data Volume', y=app_col, data=merged_df)
    plt.xlabel('Total DL+UL Data (Bytes)')
    plt.ylabel(app_col)
    plt.title(f'Relationship between Total Data Volume and {app_col}')
    plt.show()


    # Calculate and print the correlation matrix
    correlation_matrix = merged_df[['Total Data Volume', app_col]].corr()
    print(f"Correlation Matrix for {app_col}:\n{correlation_matrix}\n")


# **Interpretation-Most high degree of corrleation found with Game Download and Total Volume consumed**

# **Plot between total duration spent and total data volume consumption for top 6 MSISDN/Number**

# In[57]:



user_stats = df.groupby('MSISDN/Number').agg({
    'Dur. (ms)': 'sum',
    'Total DL (Bytes)': 'sum',
    'Total UL (Bytes)': 'sum'
}).reset_index()

# Calculate total data volume
user_stats['Total Data Volume'] = user_stats['Total DL (Bytes)'] + user_stats['Total UL (Bytes)']

# Sort by total data volume in descending order and get the top 6
top_6_users = user_stats.sort_values('Total Data Volume', ascending=False).head(6)

# Create a scatter plot with hue for MSISDN/Number
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Total Data Volume', y='Dur. (ms)', hue='MSISDN/Number', data=top_6_users, palette='viridis')
plt.xlabel('Total Data Volume (Bytes)')
plt.ylabel('Total Duration (ms)')
plt.title('Total Duration vs. Total Data Volume for Top 6 MSISDNs')
plt.legend(title='MSISDN/Number', loc='best', fontsize='small')
plt.show()


# **Variable transformations– segment the users into the top five decile classes
# based on the total duration for all sessions and compute the total data (DL+UL)
#  per decile class**

# In[59]:


# Assuming 'total_data_volume_per_bearer' DataFrame contains 'Bearer Id' and 'Total Data Volume'
total_data_volume_per_bearer['Decile Class'] = pd.qcut(total_data_volume_per_bearer['Total Data Volume'], 5, labels=False)

# Group by decile class and calculate the total data volume per class
total_data_per_decile = total_data_volume_per_bearer.groupby('Decile Class')['Total Data Volume'].sum().reset_index()

print(total_data_per_decile)


# **Dimensionality Reduction– perform a principal component analysis to reduce the dimensions of your data to 15 PCA components and provide a useful interpretation of the results**

# In[60]:



from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Select numerical features for PCA
numerical_features = df.select_dtypes(include=np.number)

# Standardize the data
x = StandardScaler().fit_transform(numerical_features)

# Apply PCA with 15 components
pca = PCA(n_components=15)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents)

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance_ratio)
print("Total Explained Variance:", np.sum(explained_variance_ratio))

# Loadings (contribution of each original feature to each principal component)
loadings = pd.DataFrame(pca.components_.T, index=numerical_features.columns, columns=[f'PC{i+1}' for i in range(15)])
print("Loadings:\n", loadings)


# **Interpretation**
# 
# **More than 60 percent variation explained by 15 PCA Components**
# 
# **PC1 has high positive loadings on "Total DL (Bytes)" and "Total UL (Bytes)", it might represent the overall data usage of the user**.
# 
# **PC2 has high positive loadings on "Social Media DL (Bytes)" and "Social Media UL (Bytes)", it might represent the social media usage of the user.**

# In[ ]:





# In[ ]:





# **Correlation Analysis– compute a correlation matrix for the following variables
#  and interpret your findings: Social Media data, Google data, Email data, YouTube  data, Netflix data, Gaming data, and Other data**

# In[61]:


# Select the relevant columns for correlation analysis
data_columns = ['Social Media DL (Bytes)', 'Social Media UL (Bytes)',
                'Google DL (Bytes)', 'Google UL (Bytes)',
                'Email DL (Bytes)', 'Email UL (Bytes)',
                'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
                'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
                'Gaming DL (Bytes)', 'Gaming UL (Bytes)']

# Create a correlation matrix for these columns
correlation_matrix = df[data_columns].corr()

# Display the correlation matrix
print(correlation_matrix)

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".4f")
plt.title('Correlation Matrix of Application Data')
plt.show()


# 

# **Part-2** **User Engagement Analysis** 

# **Session Frequency Based on  MSISDN/Number**

# In[63]:


# Assuming 'df' is your DataFrame and 'MSISDN/Number' is the column representing user sessions
sessions_frequency = df.groupby('MSISDN/Number')['MSISDN/Number'].count().sort_values(ascending=False)

sessions_frequency


# **Duration of Session**

# In[64]:


# Assuming 'df' is your DataFrame and it has 'Start' and 'End' columns representing session start and end times.

# Convert 'Start' and 'End' columns to datetime objects if they are not already
df['Start'] = pd.to_datetime(df['Start'])
df['End'] = pd.to_datetime(df['End'])

# Calculate the duration of each session
df['Session Duration'] = df['End'] - df['Start']

# Print the DataFrame with the calculated session duration
df.groupby('MSISDN/Number')['Session Duration'].value_counts().groupby(level=0).head(12)


# **thesession total traffic (download and upload (bytes))**

# In[65]:


# Assuming 'df' is your DataFrame and it has 'Total DL (Bytes)' and 'Total UL (Bytes)' columns.
total_traffic = df['Total DL (Bytes)'].sum() + df['Total UL (Bytes)'].sum()

print(f"The total traffic (download and upload) is: {total_traffic} bytes")


#  Task 2.1- Based on the above submit the Python script and slide:
#  
#  ● Aggregate the above metrics per customer ID (MSISDN) and report the top 10
#  customers per engagement metric
#  
#  ● Normalize each engagement metric and run a k-means (k=3) to classify customers into
#  three groups of engagement.
#  
#  ● Compute the minimum, maximum, average & total non-normalized metrics for each
#  cluster. Interpret your results visually with accompanying text explaining your findings.
#  
#  ● Aggregate user total traffic per application and derive the top 10 most engaged users per
#  application
#  
#  ● Plot the top 3 most used applications using appropriate charts.
#  
#  ● Using the k-means clustering algorithm, group users in k engagement clusters based on
#  the engagement metrics:
#  
#  ○ What isthe optimized value of k (use the elbow method for this)?
#  
#  ○ Interpret your findings.
#  
# 

# **Aggregate the above metrics per customer ID (MSISDN) and report the top 10 customers per engagement metric**
# 

# In[66]:


# Assuming  DataFrame is named 'df' and 'Bearer Id' represents the customer ID (MSISDN)

# Aggregate metrics per customer ID
customer_engagement = df.groupby('MSISDN/Number').agg({
    'Dur. (ms)': 'sum',  # Total session duration
    'Total DL (Bytes)': 'sum',  # Total download data
    'Total UL (Bytes)': 'sum',  # Total upload data
    'Bearer Id': 'count'  # Number of sessions
}).rename(columns={'Dur. (ms)': 'Total Session Duration', 'Bearer Id': 'Number of Sessions'})

# Report the top 10 customers per engagement metric
top_10_duration = customer_engagement.nlargest(10, 'Total Session Duration')
top_10_download = customer_engagement.nlargest(10, 'Total DL (Bytes)')
top_10_upload = customer_engagement.nlargest(10, 'Total UL (Bytes)')
top_10_sessions = customer_engagement.nlargest(10, 'Number of Sessions')

print("Top 10 Customers by Total Session Duration:\n", top_10_duration)
print("\nTop 10 Customers by Total Download Data:\n", top_10_download)
print("\nTop 10 Customers by Total Upload Data:\n", top_10_upload)
print("\nTop 10 Customers by Number of Sessions:\n", top_10_sessions)


# **Interpretation**
# - Number of session by the top three MSISDN/Number are 1066,18,17 respectively
# - Top 3 session Duration for the MSISDN Number are 
# - Total Data Down Load and upload by the top 3 MSISDN/Number are respectively

# **Normalize each engagement metric and run a k-means (k=3) to classify customers into three groups of engagement**

# In[67]:



from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# Assuming 'customer_engagement' DataFrame contains aggregated metrics per customer
engagement_metrics = customer_engagement[['Total Session Duration', 'Total DL (Bytes)', 'Total UL (Bytes)', 'Number of Sessions']]

# Normalize engagement metrics
scaler = MinMaxScaler()
normalized_engagement = scaler.fit_transform(engagement_metrics)

# K-means clustering with k=3
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(normalized_engagement)

# Add cluster labels to the DataFrame
customer_engagement['Cluster'] = clusters

# Compute statistics for each cluster
cluster_stats = customer_engagement.groupby('Cluster').agg({
    'Total Session Duration': ['min', 'max', 'mean', 'sum'],
    'Total DL (Bytes)': ['min', 'max', 'mean', 'sum'],
    'Total UL (Bytes)': ['min', 'max', 'mean', 'sum'],
    'Number of Sessions': ['min', 'max', 'mean', 'sum']
})

print("Cluster Statistics:\n", cluster_stats)

# Visualize the clusters (e.g., using scatter plots)
# ... (You can use Seaborn or Matplotlib for this)

# For example, a scatter plot of Total Session Duration vs. Total Data Volume:
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Total Session Duration', y='Total DL (Bytes)', hue='Cluster', data=customer_engagement)
plt.xlabel('Total Session Duration (ms)')
plt.ylabel('Total Download Data (Bytes)')
plt.title('Customer Clusters based on Engagement Metrics')
plt.show()

# Interpret the results visually and with accompanying text:
# ... (Analyze the cluster statistics and visualizations to understand the characteristics of each cluster)
# For example, Cluster 0 might represent high-engagement users with long sessions and high data volume,
# while Cluster 1 might represent medium-engagement users, and Cluster 2 might represent low-engagement users.


# **Compute the minimum, maximum, average & total non-normalized metrics for each cluster. Interpret your results visually with accompanying text explaining your findings**

# In[95]:


# Assuming 'customer_engagement' DataFrame contains aggregated metrics per customer and 'Cluster' column
# Compute statistics for each cluster
cluster_stats = customer_engagement.groupby('Cluster').agg({
    'Total Session Duration': ['min', 'max', 'mean', 'sum'],
    'Total DL (Bytes)': ['min', 'max', 'mean', 'sum'],
    'Total UL (Bytes)': ['min', 'max', 'mean', 'sum'],
    'Number of Sessions': ['min', 'max', 'mean', 'sum']
})

print("Cluster Statistics:\n", cluster_stats)

# Visualize the clusters (e.g., using scatter plots, box plots)
# ... (You can use Seaborn or Matplotlib for this)


# Example scatter plot of Total Session Duration vs. Total Data Volume:
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Total Session Duration', y='Total DL (Bytes)', hue='Cluster', data=customer_engagement)
plt.xlabel('Total Session Duration (ms)')
plt.ylabel('Total Download Data (Bytes)')
plt.title('Customer Clusters based on Engagement Metrics')
plt.show()


# Example box plot of Total Session Duration for each cluster:
plt.figure(figsize=(8, 6))
sns.boxplot(x='Cluster', y='Total Session Duration', data=customer_engagement)
plt.xlabel('Cluster')
plt.ylabel('Total Session Duration (ms)')
plt.title('Total Session Duration Distribution across Clusters')
plt.show()



# **Interpret the results visually and with accompanying text:
# 
# **Cluster 0 appears to represent high-engagement users, characterized by significantly longer total session durations and higher total data download volumes compared to other clusters**
# 
# **Cluster 1 represents moderate engagement, showing intermediate values for session duration and data volume**
# 
# **Cluster 2 shows the lowest engagement levels with the shortest session durations and smallest total data volumes**
# 
# **We can further analyze the characteristics of each cluster by examining the other metrics like Total UL (Bytes) and Number of Sessions**
# 
# **We can also investigate the user behavior within each cluster by examining other features in your data or conducting further analysis using other statistical methods**
# 

# **Aggregate user total traffic per application and derive the top 10 most engaged users per application**

# In[68]:


# Assuming your DataFrame is named 'df'

# Aggregate user total traffic per application
application_columns = [
    'Social Media DL (Bytes)', 'Social Media UL (Bytes)', 'Youtube DL (Bytes)',
    'Youtube UL (Bytes)', 'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
    'Google DL (Bytes)', 'Google UL (Bytes)', 'Email DL (Bytes)',
    'Email UL (Bytes)', 'Gaming DL (Bytes)', 'Gaming UL (Bytes)'
]

app_traffic_per_user = df.groupby('MSISDN/Number').agg({
    app_col: 'sum' for app_col in application_columns
}).reset_index()

# Derive the top 10 most engaged users per application
for app_col in application_columns:
  top_10_users_per_app = app_traffic_per_user.nlargest(10, app_col)
  print(f"Top 10 users for {app_col}:\n{top_10_users_per_app[[ 'MSISDN/Number', app_col]]}")
  print("-" * 30)


# **Plot the top 3 most used applications using appropriate charts. ● Using the k-means clustering algorithm, group users in k engagement clusters based on the engagement metrics**

# In[69]:


# Assuming 'app_traffic_per_user' DataFrame contains aggregated application traffic per user
# and 'application_columns' contains the list of application columns

# Plot the top 3 most used applications using bar charts
top_3_apps = app_traffic_per_user[application_columns].sum().nlargest(3)

plt.figure(figsize=(10, 6))
plt.bar(top_3_apps.index, top_3_apps.values)
plt.xlabel('Application')
plt.ylabel('Total Traffic (Bytes)')
plt.title('Top 3 Most Used Applications')
plt.xticks(rotation=45, ha='right')
plt.show()






# **Using the k-means clustering algorithm, group users in k engagement clusters based on engagement metrics**
# 
# **What is the optimized value of k (use the elbow method for this)?**

# In[70]:




# Assuming 'engagement_metrics' DataFrame contains normalized engagement metrics
inertia = []
for k in range(1, 11):
  kmeans = KMeans(n_clusters=k, random_state=42)
  kmeans.fit(engagement_metrics)
  inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()


# **Interpret your findings
# 
# **(Analyze the elbow plot to determine the optimal k value)
# 
# **For example, the elbow point might be around k=3 or k=4, indicating that 3 or 4 clusters might be a good choice for grouping users based on their engagement.
# 
# 
# **Note: You can adjust the range of k values and the visualization settings based on your specific data and needs**

# ** Interpret your findings.
# 
# Okay, let's interpret the findings from this Python code analyzing user engagement data.
# 
# **Core Objectives:**
# 
# The code aims to analyze user engagement patterns based on data usage, session duration, and application preferences. It employs techniques like scatter plots, correlation analysis, PCA, and k-means clustering to identify key insights.
# 
# **Key Interpretations and Findings:**
# 
# 1. **Data Volume and Application Usage Correlation:**
#    - The code calculates the correlation between total data volume and usage for various applications (e.g., Social Media, YouTube, Netflix).
#    - **Finding:**  The correlation analysis helps identify which applications are most strongly related to higher overall data usage. For example, if YouTube data volume shows a strong positive correlation with total data volume, it suggests that users who consume more YouTube content generally have a higher overall data usage.
# 
# 2. **Session Duration vs. Data Volume:**
#    - Scatter plots are used to analyze the relationship between total session duration and total data volume for the top 6 users (identified by highest data volume).
#    - **Finding:** The scatter plot helps reveal if there is a positive correlation between session duration and data volume, indicating that users with longer sessions generally use more data. It could also help identify outliers, where users have short sessions but very high data volume, possibly due to large file downloads.
# 
# 3. **User Segmentation with Deciles:**
#    - The code divides users into deciles based on their total data volume.
#    - **Finding:** This categorization can reveal patterns in data consumption across different user segments. For example, users in the top decile might have very different usage behavior compared to those in the lower deciles.
# 
# 4. **Principal Component Analysis (PCA):**
#    - PCA is applied to reduce the dimensionality of the data and identify underlying patterns.
#    - **Finding:** PCA helps create new features (principal components) that capture the most important variations in the data. By analyzing the explained variance ratio and loadings, you can gain insights into which original features contribute most to these new features. For example, PC1 might represent overall data usage, PC2 might represent social media usage, etc.
# 
# 5. **Correlation Matrix of Application Data:**
#    - A correlation matrix is generated to understand the relationship between different application data.
#    - **Finding:** The heatmap of the correlation matrix helps reveal how usage of different applications is linked. For example, if Social Media data and YouTube data have a strong positive correlation, it suggests users who heavily use social media also tend to consume more YouTube content.
# 
# 6. **Session Frequency Analysis:**
#    - The code calculates the frequency of sessions for each user (Bearer ID).
#    - **Finding:** This helps identify the users with the highest number of sessions, which could be considered highly active or engaged users.
# 
# 7. **Total Traffic Calculation:**
#    - The code calculates the total traffic (upload + download) for all users.
#    - **Finding:** This provides a general overview of the network's total data usage.
# 
# 8. **Customer Engagement Analysis:**
#    - **Aggregation per Customer:** Metrics like session duration, data volume, and session counts are aggregated per customer (MSISDN).
#    - **Top 10 Customers:**  The top 10 customers based on various engagement metrics are identified.
#    - **Normalization and Clustering:** Engagement metrics are normalized, and k-means clustering (with k=3) is used to group customers into clusters based on their engagement level.
#    - **Cluster Statistics and Interpretation:**  Statistics like the minimum, maximum, average, and total values are computed for each cluster.  Visualizations (e.g., scatter plots) help understand the characteristics of each cluster.
#    - **Finding:**  The clusters reveal insights into user behavior. For example, one cluster might represent high-engagement users, another represents medium-engagement users, and a third represents low-engagement users.
# 
# 9. **Application Usage by Users:**
#    - User total traffic per application is calculated.
#    - **Top 10 Users:** The top 10 most engaged users for each application are identified.
#    - **Top Application Usage:** The top 3 most used applications are plotted using a bar chart.
#    - **Finding:** This helps understand which applications are the most popular among users and which users are the most engaged with each application.
# 
# 10. **Optimal k for K-Means Clustering:**
#     - The elbow method is used to find the optimal number of clusters (k) for k-means.
#     - **Finding:** The elbow point on the inertia plot suggests the optimal k value, which represents a balance between maximizing variance explained within clusters and minimizing the number of clusters.
# 
# 
# **Overall:**
# 
# This code provides a comprehensive analysis of user engagement. It combines descriptive statistics, correlation analysis, dimensionality reduction, and clustering to understand patterns in data usage, application preference, and user activity. The analysis helps identify key characteristics of high and low engagement users, most popular applications, and the general traffic load on the network.
# 
# The insights from this analysis can be valuable for network optimization, personalized user experience enhancements, and identifying opportunities for marketing campaigns and targeted promotions.
# 

# **Task 3. 1- Aggregate, per customer, the following information (treat missing & outliers by
# replacing with the mean or the mode of the corresponding variable)** :
# 
# **Average TCP retransmission
# 
# **Average RTT
# 
# **Handset type
# 
# **Average throughput

# In[76]:


df['TCP Retransmission']=df['TCP DL Retrans. Vol (Bytes)']+df['TCP UL Retrans. Vol (Bytes)']
df['RTT']=df['Avg RTT UL (ms)']+df['Avg RTT DL (ms)']
df['Throughput']=df['Avg Bearer TP DL (kbps)']+df['Avg Bearer TP UL (kbps)']


# In[77]:


def aggregate_customer_data(df):
  """
  Aggregates customer data, handling missing and outlier values.

  Args:
    df: DataFrame containing customer data.

  Returns:
    DataFrame with aggregated customer information.
  """

  # Replace missing values with the mean or mode
  df['TCP Retransmission'].fillna(df['TCP Retransmission'].mean(), inplace=True)
  df['RTT'].fillna(df['RTT'].mean(), inplace=True)
  df['Handset Type'].fillna(df['Handset Type'].mode()[0], inplace=True)
  df['Throughput'].fillna(df['Throughput'].mean(), inplace=True)

  # Replace outliers with the mean
  for column in ['TCP Retransmission', 'RTT', 'Throughput']:
      q1 = df[column].quantile(0.25)
      q3 = df[column].quantile(0.75)
      iqr = q3 - q1
      lower_bound = q1 - 1.5 * iqr
      upper_bound = q3 + 1.5 * iqr
      df[column] = np.where((df[column] < lower_bound) | (df[column] > upper_bound),
                            df[column].mean(), df[column])

  # Aggregate per customer
  aggregated_df = df.groupby('MSISDN/Number').agg({
      'TCP Retransmission': 'mean',
      'RTT': 'mean',
      'Handset Type': lambda x: x.mode()[0],  # Mode for Handset Type
      'Throughput': 'mean'
  }).reset_index()

  return aggregated_df



# Example usage (assuming 'df' is your DataFrame):
aggregated_customer_df = aggregate_customer_data(df)
print(aggregated_customer_df)


# **
# **Task 3.2- Compute & list 10 of the top, bottom, and most frequent:**
# 
# **a. TCPvalues in the dataset.**
# 
# **b. RTTvalues in the dataset.**
# 
# **c. Throughput values in the dataset.**

# In[78]:




def compute_top_bottom_frequent(df, column_name, n=10):
  """
  Computes the top, bottom, and most frequent values for a given column in a DataFrame.

  Args:
    df: DataFrame containing the data.
    column_name: Name of the column to analyze.
    n: Number of top/bottom/frequent values to return.

  Returns:
    A tuple containing the top n, bottom n, and most frequent n values.
  """
  #df['TCP Retransmission']=df['TCP DL Retrans. Vol (Bytes)']+df['TCP UL Retrans. Vol (Bytes)']
  #df['RTT']=df['Avg RTT UL (ms)']+df['Avg RTT DL (ms)']
  #df['Throughput']=df['Avg Bearer TP DL (kbps)']+df['Avg Bearer TP UL (kbps)']

  top_n = df['TCP Retransmission'].nlargest(n).tolist()
  bottom_n = df['TCP Retransmission'].nsmallest(n).tolist()

  # Calculate value frequencies
  value_counts = df['TCP Retransmission'].value_counts()
  most_frequent_n = value_counts.nlargest(n).index.tolist()

  return top_n, bottom_n, most_frequent_n


# Assuming 'df' is your DataFrame and it has 'TCP Retransmission', 'RTT', and 'Throughput' columns.

# a. TCP Values
top_tcp, bottom_tcp, frequent_tcp = compute_top_bottom_frequent(df, 'TCP Retransmission')
print("Top 10 TCP Values:", top_tcp)
print("Bottom 10 TCP Values:", bottom_tcp)
print("Most Frequent 10 TCP Values:", frequent_tcp)


# b. RTT Values
top_rtt, bottom_rtt, frequent_rtt = compute_top_bottom_frequent(df, 'RTT')
print("\nTop 10 RTT Values:", top_rtt)
print("Bottom 10 RTT Values:", bottom_rtt)
print("Most Frequent 10 RTT Values:", frequent_rtt)


# c. Throughput Values
top_throughput, bottom_throughput, frequent_throughput = compute_top_bottom_frequent(df, 'Throughput')
print("\nTop 10 Throughput Values:", top_throughput)
print("Bottom 10 Throughput Values:", bottom_throughput)
print("Most Frequent 10 Throughput Values:", frequent_throughput)


# **Task 3.3- Compute & report:**
# 
# **d. The distribution of the average throughput per handset type and provide**
# 
# **interpretation for your findings.**
# 
# **e. The average TCP retransmission view per handset type and provide**
# 
# **interpretation for your findings.**

# In[79]:




def analyze_throughput_retransmission(df):
  """
  Analyzes the distribution of average throughput and TCP retransmission per handset type.

  Args:
    df: DataFrame containing the data.

  Returns:
    None. Prints the analysis results.
  """

  # Group data by Handset Type and calculate mean throughput and TCP retransmission
  handset_stats = df.groupby('Handset Type').agg({
      'Throughput': 'mean',
      'TCP Retransmission': 'mean'
  })

  # Print the distribution of average throughput per handset type
  print("\nAverage Throughput per Handset Type:")
  print(handset_stats['Throughput'])

  # Provide interpretation for the throughput findings.
  print("\nInterpretation of Throughput:")
  # Example: If certain handset types consistently show higher average throughput, it might indicate better network compatibility or more efficient data handling for those devices.
  # Example: You can also check if there are outliers for any handset type and investigate the possible reasons.


  # Print the average TCP retransmission per handset type
  print("\nAverage TCP Retransmission per Handset Type:")
  print(handset_stats['TCP Retransmission'])

  # Provide interpretation for the TCP retransmission findings.
  print("\nInterpretation of TCP Retransmission:")
  # Example: Higher TCP retransmission rates for specific handset types might suggest network instability, congestion, or potential issues with the devices themselves.
  # Example: You can investigate if there's a correlation between retransmission rates and other factors like network location or time of day.




# Assuming 'df' is your DataFrame and it has 'Handset Type', 'Throughput', and 'TCP Retransmission' columns.
# Call the function to perform the analysis.
analyze_throughput_retransmission(df)


# **3.4- Using the experience metrics above, perform a k-means clustering (where k = 3) to**
# 
# **segment users into groups of experiences and provide a brief description of each cluster. (The**
# 
# **description must define each group based on your understanding of the data)**
# 
# 
# **Assuming 'engagement_metrics' DataFrame contains normalized engagement metrics**

# In[82]:



kmeans = KMeans(n_clusters=3, random_state=42)  # k=3 for 3 clusters
kmeans.fit(engagement_metrics)
customer_engagement['Cluster'] = kmeans.labels_

# Compute statistics for each cluster
cluster_stats = customer_engagement.groupby('Cluster').agg({
    'Total Session Duration': ['mean', 'std'],
    'Total DL (Bytes)': ['mean', 'std'],
    'Total UL (Bytes)': ['mean', 'std'],
    'Number of Sessions': ['mean', 'std']
})

print("Cluster Statistics:\n", cluster_stats)

# Brief description of each cluster based on the computed statistics:

# Example:
print("\nCluster Descriptions:")
print("Cluster 0: Low Engagement - Characterized by short average session durations, low average data download and upload volumes, and fewer sessions.")
print("Cluster 1: Medium Engagement - Shows moderate session durations, data volumes, and session numbers, representing a balance between high and low engagement.")
print("Cluster 2: High Engagement -  Distinguished by long average session durations, high data download and upload volumes, and a greater number of sessions.")

# Visualize the clusters using scatter plots, box plots, etc.
# (You've already done this earlier in your code using Seaborn and Matplotlib)

# Note: You can provide a more accurate and insightful description of each cluster by examining the specific values
# in cluster_stats and potentially using further analysis techniques like examining the distribution of other features
# within each cluster.


#  **Write a Python program to assign:**
#  
# **4.a. engagement score to each user. Consider the engagement score as the**
# 
# **Euclidean distance between the user data point & the less engaged cluster (use**
# 
# **the first clustering for this)**

# In[83]:




# Assuming 'customer_engagement' DataFrame contains user data with engagement metrics
# and 'kmeans' is the fitted KMeans model from your previous code

# Calculate Euclidean distance between each user and the less engaged cluster centroid
less_engaged_centroid = kmeans.cluster_centers_[0]  # Assuming cluster 0 represents the less engaged group
user_engagement_scores = []
for index, row in customer_engagement.iterrows():
    user_data = row[['Total Session Duration', 'Total DL (Bytes)', 'Total UL (Bytes)', 'Number of Sessions']]
    euclidean_distance = np.linalg.norm(user_data - less_engaged_centroid)
    user_engagement_scores.append(euclidean_distance)

# Assign engagement scores to each user
customer_engagement['Engagement Score'] = user_engagement_scores

# Print the DataFrame with the added 'Engagement Score' column
customer_engagement


# **TBC**

# In[87]:


# prompt: experience score for each user. Consider the experience score as the Euclidean
#  distance between the user data point & the worst experience cluster.

# Assuming 'customer_engagement' DataFrame contains user data with engagement metrics
# and 'kmeans' is the fitted KMeans model from your previous code

# Find the centroid of the cluster with the lowest average engagement
cluster_centroids = kmeans.cluster_centers_
average_engagement_per_cluster = customer_engagement.groupby('Cluster')['Total Session Duration'].mean()
least_engaged_cluster = average_engagement_per_cluster.idxmin()
worst_experience_centroid = cluster_centroids[least_engaged_cluster]

# Calculate Euclidean distance between each user and the worst experience cluster centroid
user_experience_scores = []
for index, row in customer_engagement.iterrows():
    user_data = row[['Total Session Duration', 'Total DL (Bytes)', 'Total UL (Bytes)', 'Number of Sessions']]
    euclidean_distance = np.linalg.norm(user_data - worst_experience_centroid)
    user_experience_scores.append(euclidean_distance)

# Assign experience scores to each user
customer_engagement['Experience Score'] = user_experience_scores

# Print the DataFrame with the added 'Experience Score' column
customer_engagement


# In[88]:


# prompt: Task 4.2- Consider the average of both engagement & experience scores as the satisfaction
#  score & report the top 10 satisfied customer

# Calculate the average of Engagement Score and Experience Score as Satisfaction Score
customer_engagement['Satisfaction Score'] = (customer_engagement['Engagement Score'] + customer_engagement['Experience Score']) / 2

# Sort the DataFrame by Satisfaction Score in descending order and get the top 10 satisfied customers
top_10_satisfied_customers = customer_engagement.sort_values('Satisfaction Score', ascending=False).head(10)

# Print the top 10 satisfied customers
top_10_satisfied_customers


# In[91]:


# prompt:  Task 4.3- Build a regression model of your choice to predict the satisfaction score of a
#  customer. linear and XGBOOST, find the Mean square error and r2 value for both the models

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Assuming 'customer_engagement' DataFrame contains your data with 'Satisfaction Score' as the target variable
# and other relevant features.

# Separate features (X) and target (y)
X = customer_engagement.drop('Satisfaction Score', axis=1)
y = customer_engagement['Satisfaction Score']

# Convert categorical features to numerical using one-hot encoding or label encoding if necessary
# For example:
# X = pd.get_dummies(X, columns=['Handset Type'], drop_first=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Calculate MSE and R2 for Linear Regression
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print("Linear Regression:")
print("Mean Squared Error:", mse_linear)
print("R-squared:", r2_linear)

# 2. XGBoost Model
xgb_model = XGBRegressor()
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# Calculate MSE and R2 for XGBoost
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print("\nXGBoost:")
print("Mean Squared Error:", mse_xgb)
print("R-squared:", r2_xgb)


# In[92]:


# prompt: Run a k-means (k=2) on the engagement & the experience score

# Assuming 'customer_engagement' DataFrame contains 'Engagement Score' and 'Experience Score'
X = customer_engagement[['Engagement Score', 'Experience Score']]

# Apply K-Means with k=2
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# Add cluster labels to the DataFrame
customer_engagement['Cluster'] = kmeans.labels_

# Analyze the clusters (optional)
# You can compute cluster statistics, visualize the clusters, etc.
# Example:
# print(customer_engagement.groupby('Cluster').agg({'Engagement Score': 'mean', 'Experience Score': 'mean'}))

# Visualize the clusters using a scatter plot
plt.scatter(customer_engagement['Engagement Score'], customer_engagement['Experience Score'], c=customer_engagement['Cluster'])
plt.xlabel('Engagement Score')
plt.ylabel('Experience Score')
plt.title('K-Means Clustering (k=2)')
plt.show()


# In[94]:


# prompt: Aggregate the average satisfaction & experience score per cluster.

# Assuming 'customer_engagement' DataFrame contains 'Cluster', 'Engagement Score', and 'Experience Score' columns.

cluster_satisfaction = customer_engagement.groupby('Cluster').agg({
    'Engagement Score': 'mean',
    'Experience Score': 'mean'
})

print("Average Satisfaction & Experience Score per Cluster:\n", cluster_satisfaction)


# In[96]:


get_ipython().system('pip install mysql-connector-python')


# In[102]:


# prompt:  Export your final table containing all user ID + engagement, experience &
#  satisfaction scores in your local MySQL database. Report a screenshot of a select query output
#  on the exported table.

import mysql.connector

# Replace with your database credentials
mydb = mysql.connector.connect(
  host="Mysql@localhost:3306",
  user="root",
  password="root",
  database="your_database"
)

mycursor = mydb.cursor()

# Assuming customer_engagement DataFrame is defined
for index, row in customer_engagement.iterrows():
    user_id = row['MSISDN/Number']  # Assuming 'MSISDN/Number' is your user ID column
    engagement_score = row['Engagement Score']
    experience_score = row['Experience Score']
    satisfaction_score = row['Satisfaction Score']
    sql = "INSERT INTO customer_scores (user_id, engagement_score, experience_score, satisfaction_score) VALUES (%s, %s, %s, %s)"
    val = (user_id, engagement_score, experience_score, satisfaction_score)
    mycursor.execute(sql, val)

mydb.commit()



# Screenshot of the select query output will be manually captured after running this code.


# In[ ]:


print(mycursor.rowcount, "was inserted.")

# Select query for the exported table (replace your_database and customer_scores with your table details)
sql = "SELECT * FROM your_database.customer_scores"
mycursor.execute(sql)
myresult = mycursor.fetchall()

for x in myresult:
x


# In[ ]:





# In[ ]:





# In[ ]:




