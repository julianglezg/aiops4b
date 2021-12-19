import pandas as pd

# Raw Data - we import the datasets from the Brazilian E-Commerce Dataset

orders = pd.read_csv(r'olist_orders_dataset.csv')

payments = pd.read_csv(r'olist_order_payments_dataset.csv')
payments['payment_id'] = range(0, 103886)

reviews = pd.read_csv(r'olist_order_reviews_dataset.csv')

items = pd.read_csv(r'olist_order_items_dataset.csv')

customers = pd.read_csv(r'olist_customers_dataset.csv')

geolocation = pd.read_csv(r'olist_geolocation_dataset.csv')

# METRICS - we will now create the different metrics

# Date

date = orders['order_purchase_timestamp']
date = date.sort_values(ascending=True)
timestamp = date
date = date.str.slice(stop=10)
date = date.unique()

# Daily total revenue

df = pd.merge(orders, payments, on='order_id')
df['order_purchase_timestamp'] = df['order_purchase_timestamp'].str.slice(stop=10)
df = df.sort_values(by='order_purchase_timestamp')
rev = df[['order_purchase_timestamp', 'payment_value']]
revenue = rev.groupby(['order_purchase_timestamp']).sum()
###print(revenue.head())

# STD Total Customers

rev_std = rev.groupby('order_purchase_timestamp').agg(['std'])
###print(rev_std.head(15))

# STD New Customers

df = pd.merge(orders, customers, on='customer_id')
df['new_customers_orders'] = 1
df['returning_customers_orders'] = 1
df['total_customer_orders'] = 1
df['order_purchase_timestamp'] = df['order_purchase_timestamp'].str.slice(stop=10)
df = df.sort_values(by='order_purchase_timestamp')
# print(df.head())

df2 = df.duplicated('customer_unique_id', 'first')
# print(df2.tail())
customer_list = list(df2)
new_customers = [i for i, j in enumerate(customer_list) if not j]

new_customers_df = df.loc[df.index[new_customers]]
new_customers_df2 = pd.merge(new_customers_df,payments,on='order_id')
nco = new_customers_df2[['order_purchase_timestamp', 'payment_value']]
rev_new_std = nco.groupby('order_purchase_timestamp').agg(['count','sum'])
###print(rev_new_std.head(15))

## Daily returning customers orders ##

pd.options.mode.chained_assignment = None

returning_customers = [i for i, j in enumerate(customer_list) if j]

returning_customers_df = df.loc[df.index[returning_customers]]
returning_customers_df2 = pd.merge(returning_customers_df,payments,on='order_id')
rco = returning_customers_df2[['order_purchase_timestamp', 'payment_value']]
rev_ret_std = rco.groupby('order_purchase_timestamp').agg(['count','sum'])
###print(rev_ret_std.head(15))

ratio = pd.merge(rev_ret_std,rev_new_std,on='order_purchase_timestamp')
ratio['returning/new_customers_ratio'] = ratio['payment_value_x']['count']/ratio['payment_value_y']['count']
ratio = ratio[['order_purchase_timestamp','returning/new_customers_ratio']]
#print(ratio.head())


# Total customer orders

df = pd.merge(orders, customers, on='customer_id')
df['new_customers_orders'] = 1
df['returning_customers_orders'] = 1
df['total_customer_orders'] = 1
df['order_purchase_timestamp'] = df['order_purchase_timestamp'].str.slice(stop=10)
df = df.sort_values(by='order_purchase_timestamp')
# print(df.head())


# Total daily customer orders

tco = df[['order_purchase_timestamp', 'total_customer_orders']]
total_customer_orders = tco.groupby(['order_purchase_timestamp']).sum()
#print(total_customer_orders.head(10))



## Average/STD daily product popularity ##

pp = pd.merge(orders, reviews, on='order_id')
pp['order_purchase_timestamp'] = pp['order_purchase_timestamp'].str.slice(stop=10)
pp2 = pp[['order_purchase_timestamp', 'review_score']]
avg_review_score = pp2.groupby(['order_purchase_timestamp']).agg(['mean'])
###print(product_popularity.head(10))

# Total good/bad reviews #

good = []
bad = []

for i in pp['review_score']:
    if i>=3:
        good.append(1)
    else:
        good.append(0)

for i in pp['review_score']:
    if i<3:
        bad.append(1)
    else:
        bad.append(0)

pp['good_review'] = good
pp['bad_review'] = bad

pp3 = pp[['order_purchase_timestamp', 'good_review','bad_review']]
good_bad_reviews = pp3.groupby(['order_purchase_timestamp']).agg(['sum'])
##print(good_bad_reviews.head(10))


## Average/STD delivery value ##

afv = pd.merge(orders, items, on='order_id')
afv['order_purchase_timestamp'] = afv['order_purchase_timestamp'].str.slice(stop=10)
afv = afv[['order_purchase_timestamp', 'freight_value']]
afv_total = afv.groupby(['order_purchase_timestamp']).agg(['mean'])
###print(afv_total.head(10))

## Average/STD delivery delay ## we are not going to use this

delivery = orders.copy()
delivery['order_purchase_timestamp'] = delivery['order_purchase_timestamp'].str.slice(stop=10)
delivery['order_delivered_customer_date'] = delivery['order_delivered_customer_date'].str.slice(stop=10)
delivery['order_estimated_delivery_date'] = delivery['order_estimated_delivery_date'].str.slice(stop=10)

delivery['order_delivered_customer_date'] = pd.to_datetime(delivery['order_delivered_customer_date'])
delivery['order_estimated_delivery_date'] = pd.to_datetime(delivery['order_estimated_delivery_date'])

delivery['delivery_delay'] = delivery['order_delivered_customer_date'] - delivery['order_estimated_delivery_date']

delay = delivery[['order_purchase_timestamp', 'delivery_delay']]
delay = delay.sort_values(by='order_purchase_timestamp')
delay = delay.dropna()
#print(delay.head(15))

score = []

# The score is 1 if there is no delay and -1 is there is any delay

for i in delay['delivery_delay']:
    if i<=pd.Timedelta(0):
        score.append(1)
    else:
        score.append(-1)

delay['delivery_score'] = score

delivery_score = delay[['order_purchase_timestamp', 'delivery_score']]
delivery_score = delivery_score.groupby(['order_purchase_timestamp']).agg(['mean'])




### METRICS DATAFRAMES ###

dataframes = [revenue, rev_new_std, rev_ret_std, avg_review_score, good_bad_reviews, ratio,
              afv_total, delivery_score, total_customer_orders]

from functools import reduce

metrics = reduce(lambda left, right: pd.merge(left, right, on=['order_purchase_timestamp'],
                                                how='outer'), dataframes)

metrics.reset_index(inplace=True)
#print(metrics.columns)

col_names = ['Time_col','Value','Total_New_Customers','Total_New_Customers_Value',
             'Total_Returning_Customers',
             'Total_Returning_Customers_Value', 'AVG_Review_Score',
             'Total_Good_Reviews','Total_Bad_Reviews','Ratio',
             'AVG_Freight_Value',
             'AVG_Delivery_Score','total_customer_orders']

metrics.columns = col_names
#metrics = metrics.iloc[14:-1,:]
metrics.reset_index(inplace=True)
metrics = metrics.drop(['index'],axis=1)


### FEATURE IMPORTANCE ### we use OLS to find the most relevant metrics

from sklearn.linear_model import LinearRegression

X = metrics[[
             'Total_New_Customers',
             'Total_Returning_Customers',
             'AVG_Review_Score',
             'Total_Good_Reviews',
             'Total_Bad_Reviews',
             'Ratio',
             'AVG_Freight_Value',
             'AVG_Delivery_Score'
             ]]

X = X.iloc[:601,:]

y = metrics[['Value']]
y = y.iloc[:601,:]

X = X.fillna(0)

import numpy as np
from scipy import stats
import statsmodels.api as sm

est = sm.OLS(y, X)
est2 = est.fit()
print(est2.summary())


## Final metrics dataset ##

from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")

metrics_pvalue = metrics[['Time_col','Value','Total_Good_Reviews','Total_Bad_Reviews','AVG_Review_Score','AVG_Freight_Value','AVG_Delivery_Score','total_customer_orders']]

metrics_pvalue.to_csv('metrics_pvalue_ord.csv')