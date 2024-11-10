**Overview**

 An investor is interested to purchase a Telco based out in Republic of Pefkakia. This investor’s due diligence on all purchases includes a detailed analysis of the data that underlies the business, to try to understand the fundamentals of the business and especially to identify opportunities to drive profitability by changing the focus of which products or services are being offered.

Telco has deployed Consultant to analyze the System generated data and find insights that will help the investor to take a decision on purchase of the Telco.

Objective:

 To analyze a telecommunication dataset that contains useful information about the customers & their activities on the network. 

To deliver insights that helps the Investor on the below four parameters :
							
 User Overview Analysis 
User Experience Analysis
User Satisfaction Analysis
 User Engagement analysis 
 **User Overview**

 ![image](https://github.com/user-attachments/assets/aa660c24-08be-45e9-87e4-34b42578a27c)

 ![image](https://github.com/user-attachments/assets/7a2cf8ab-b262-4017-8c3f-7e1e5a6c060d)
 ![image](https://github.com/user-attachments/assets/b2e4f465-6f11-48bd-bd0e-4404e5a2e108)

 Interpretation:

Top 3 MSISDN/Number based on number of XDR Sessions are 4.1882,3.362632e+10,3.362578e+10  respectively from fig-1

Total session Duration for the Top 3 MSISDN are 7.255100e+07,1 .855375e+07,9.966898e+06  respectively from fig-2

Total Data Volume by top 3 MSISDN/Number are  5.317447e+11,8.846226e+09,8.514774e+09 respectively from fig-3

Top 5 Handset manufacturers present in data set are Apple, Samsung, Huawei, undefined and  Sony from fig-4

Huawei B-5285 handset is used by maximum number of users from Fig-5

![image](https://github.com/user-attachments/assets/654d2d92-d783-4558-94b3-6b059c0d3722)


![image](https://github.com/user-attachments/assets/b37a4584-f5f3-4438-92fd-73bd90c69138)

![image](https://github.com/user-attachments/assets/df8e624f-7564-410d-a686-73d4347e1d53)


**User Engagement Analysis**

![image](https://github.com/user-attachments/assets/a6d53d64-7cb7-412f-a1fc-90ec9cfa7e8a)


![image](https://github.com/user-attachments/assets/7b4d83cb-94f3-46d1-93b1-35e6d5c7210f)




![image](https://github.com/user-attachments/assets/1692717e-9124-4997-83a9-efac08c635fc)


**Key Interpretations and Findings:**

Data Volume and Application Usage Correlation:

The code calculates the correlation between total data volume and usage for various applications (e.g., Social Media, YouTube, Netflix).
Finding: The correlation analysis helps identify which applications are most strongly related to higher overall data usage. For example, if YouTube data volume shows a strong positive correlation with total data volume, it suggests that users who consume more YouTube content generally have a higher overall data usage.

Session Duration vs. Data Volume:
Scatter plots are used to analyze the relationship between total session duration and total data volume for the top 6 users (identified by highest data volume).
Finding: The scatter plot helps reveal if there is a positive correlation between session duration and data volume, indicating that users with longer sessions generally use more data. It could also help identify outliers, where users have short sessions but very high data volume, possibly due to large file downloads.

User Segmentation with Deciles:
The code divides users into deciles based on their total data volume.
Finding: This categorization can reveal patterns in data consumption across different user segments. For example, users in the top decile might have very different usage behavior compared to those in the lower deciles.

Principal Component Analysis (PCA):
PCA is applied to reduce the dimensionality of the data and identify underlying patterns.
Finding: PCA helps create new features (principal components) that capture the most important variations in the data. By analyzing the explained variance ratio and loadings, you can gain insights into which original features contribute most to these new features. For example, PC1 might represent overall data usage, PC2 might represent social media usage, etc.

Correlation Matrix of Application Data:
A correlation matrix is generated to understand the relationship between different application data.
Finding: The heatmap of the correlation matrix helps reveal how usage of different applications is linked. For example, if Social Media data and YouTube data have a strong positive correlation, it suggests users who heavily use social media also tend to consume more YouTube content.

Session Frequency Analysis:
The code calculates the frequency of sessions for each user (MSISDN).
Finding: This helps identify the users with the highest number of sessions, which could be considered highly active or engaged users.

Total Traffic Calculation:
The code calculates the total traffic (upload + download) for all users.
Finding: This provides a general overview of the network's total data usage.

Customer Engagement Analysis:
Aggregation per Customer: Metrics like session duration, data volume, and session counts are aggregated per customer (MSISDN).

Top 10 Customers: The top 10 customers based on various engagement metrics are identified.

Normalization and Clustering: Engagement metrics are normalized, and k-means clustering (with k=3) is used to group customers into clusters based on their engagement level.

Cluster Statistics and Interpretation: Statistics like the minimum, maximum, average, and total values are computed for each cluster. Visualizations (e.g., scatter plots) help understand the characteristics of each cluster.

Finding: The clusters reveal insights into user behavior. For example, one cluster might represent high-engagement users, another represents medium-engagement users, and a third represents low-engagement users.
Application Usage by Users:

User total traffic per application is calculated.

Top 10 Users: The top 10 most engaged users for each application are identified.

Top Application Usage: The top 3 most used applications are plotted using a bar chart.

Finding: This helps understand which applications are the most popular among users and which users are the most engaged with each application.

Optimal k for K-Means Clustering:

The elbow method is used to find the optimal number of clusters (3) for k-means.
Finding: The elbow point on the inertia plot suggests the optimal k value, which represents a balance between maximizing variance explained within clusters and minimizing the number of clusters.


![image](https://github.com/user-attachments/assets/aa5f4b82-f55d-4660-b634-8238e2a3cc7c)













