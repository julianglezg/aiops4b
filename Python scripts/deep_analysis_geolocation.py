# Adapted from https://www.kaggle.com/hoonkeng/eda-understand-brazil-e-commerce-geographically

# We start by importing the required libraries

import numpy as np
import pandas as pd
import warnings
warnings.simplefilter("ignore")
import matplotlib.pyplot as plt
import plotly
import plotly.offline as offline
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.graph_objs import Scatter, Figure, Layout
from plotly import tools

# We import the datasets that we will work with, which are the geolocation, payments, order items, orders, and customers

geo = pd.read_csv(r'olist_geolocation_dataset.csv',
                  dtype={'geolocation_zip_code_prefix': str})

df_payment = pd.read_csv(r'olist_order_payments_dataset.csv')
df_items = pd.read_csv(r'olist_order_items_dataset.csv')
df_orders = pd.read_csv(r'olist_orders_dataset.csv')
df_customers = pd.read_csv(r'olist_customers_dataset.csv')

# We can merge the first two on order_id and the customers one on customer_id

df = pd.merge(df_payment,
              df_items[['order_id','price','freight_value']],
              on='order_id')

df = pd.merge(df,
              df_orders[['order_id','customer_id','order_purchase_timestamp','order_approved_at',
                         'order_delivered_customer_date','order_estimated_delivery_date']],
              on='order_id')

df = pd.merge(df,
              df_customers[['customer_id','customer_state','customer_city','customer_zip_code_prefix']],
              on='customer_id')

#print(df.head())

# We can plot the orders on the map of Brazil

data = [go.Scattermapbox(
    lon = geo['geolocation_lng'],
    lat = geo['geolocation_lat'],
    marker = dict(
        size = 5,
        color = 'green',
    ))]

layout = dict(
    title = 'Olist Orders',
    mapbox = dict(
        accesstoken = 'pk.eyJ1IjoiaG9vbmtlbmc5MyIsImEiOiJjam43cGhpNng2ZmpxM3JxY3Z4ODl2NWo3In0.SGRvJlToMtgRxw9ZWzPFrA',
        center= dict(lat=-22,lon=-43),
        bearing=10,
        pitch=0,
        zoom=2,
    )
)
fig = dict( data=data, layout=layout )
#plot(fig, validate=False)

# Now we can complete the dataframe with the correct geolocation data

df['customer_state'] = df['customer_state'].apply(lambda x : x.lower())
df['customer_city'] = df['customer_city'].apply(lambda x : x.lower())

geo_state = geo.groupby('geolocation_state')['geolocation_lat','geolocation_lng'].mean().reset_index()
geo_state['geolocation_state'] = geo_state['geolocation_state'].apply(lambda x : x.lower())

geo_city = geo.groupby('geolocation_city')['geolocation_lat','geolocation_lng'].mean().reset_index()
geo_city['geolocation_city'] = geo_city['geolocation_city'].apply(lambda x : x.lower())

geo_city.rename(columns={'geolocation_lat': 'c_lat','geolocation_lng':'c_lng'}, inplace=True)

df = pd.merge(df, geo_state, how='left', left_on='customer_state',right_on='geolocation_state')
df = pd.merge(df, geo_city,how='left',left_on='customer_city',right_on='geolocation_city')

#print(df.head())

# The following map shows the orders with the cities they were made from

data = [go.Scattermapbox(
    lon = geo_city['c_lng'],
    lat = geo_city['c_lat'],
    text = geo_city['geolocation_city'],
    marker = dict(
        size = 2,
        color = 'Green',
    ))]

layout = dict(
    title = 'Olist Orders Cities',
    mapbox = dict(
        accesstoken = 'pk.eyJ1IjoiaG9vbmtlbmc5MyIsImEiOiJjam43cGhpNng2ZmpxM3JxY3Z4ODl2NWo3In0.SGRvJlToMtgRxw9ZWzPFrA',
        center= dict(lat=-22,lon=-43),
        bearing=10,
        pitch=0,
        zoom=2,
    )
)
fig = dict(data=data,layout=layout)
#plot(fig,validate=False)

# Now we can start with the analysis for the freight value

city_spend = df.groupby(['customer_city','c_lng','c_lat'])['price'].sum().to_frame().reset_index()
city_freight = df.groupby(['customer_city','c_lng','c_lat'])['freight_value'].mean().reset_index()

state_spend = df.groupby(['customer_state','c_lng','c_lat'])['price'].sum().to_frame().reset_index()
state_freight = df.groupby(['customer_state','c_lng','c_lat'])['freight_value'].mean().reset_index()
state_freight['text'] = 'state :' + state_freight['customer_state'] + ' | Freight: ' + state_freight['freight_value'].astype(str)

data = [go.Scattergeo(
    lon = state_spend['c_lng'],
    lat = state_spend['c_lat'],
    text = state_freight['text'],
    marker = dict(
        size = state_spend['price']/3000,
        sizemin = 5,
        color= state_freight['freight_value'],
        colorscale= 'Reds',
        cmin = 20,
        cmax = 50,
        line = dict(width=0.1, color='rgb(40,40,40)'),
        sizemode = 'area'
    ),
    name = 'State'),
    go.Scattergeo(
        lon = city_spend['c_lng'],
        lat = city_spend['c_lat'],
        text = city_freight['freight_value'],
        marker = dict(
            size = city_spend['price']/1000,
            sizemin = 2,
            color= city_freight['freight_value'],
            colorscale= 'Blues',
            reversescale=True,
            cmin = 0,
            cmax = 80,
            line = dict(width=0.1, color='rgb(40,40,40)'),
            sizemode = 'area'
        ),
        name = 'City')]

layout = dict(
    title = 'Olist Orders Freight Value',
    showlegend = True,
    autosize=True,
    width = 900,
    height = 600,
    geo = dict(
        scope = "south america",
        projection = dict(type='winkel tripel', scale = 1.6),
        center = dict(lon=-47,lat=-22),
        showland = True,
        showcountries= True,
        showsubunits=True,
        landcolor = 'rgb(155, 155, 155)',
        subunitwidth=1,
        countrywidth=1,
        subunitcolor="rgb(255, 255, 255)",
        countrycolor="rgb(255, 255, 255)"
    )
)

fig = dict(data=data,layout=layout)
plot(fig,validate=False)

# The North and North-Eastern regions have a higher freight value

# Now we can see the analysis for the delivery time

df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['order_approved_at'] = pd.to_datetime(df['order_approved_at'])

df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])
df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])

df['delay'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.total_seconds() / (3600 * 24)
df['deliver'] = (df['order_delivered_customer_date'] - df['order_approved_at']).dt.total_seconds() / (3600 * 24)

df['delay'] = df['delay'].fillna(0)
df['deliver'] = df['deliver'].fillna(0)

city_deliver = df.groupby(['customer_city','c_lng','c_lat'])['deliver'].mean().reset_index()
city_delay = df.groupby(['customer_city','c_lng','c_lat'])['delay'].mean().reset_index()

state_deliver = df.groupby(['customer_state','c_lng','c_lat'])['deliver'].mean().reset_index()
state_delay = df.groupby(['customer_state','c_lng','c_lat'])['delay'].mean().reset_index()

state_deliver['text'] = 'Deliver duration: ' + state_deliver['deliver'].astype(str) + '| Delay:' + state_delay['delay'].astype(str)
city_deliver['text'] = 'Deliver duration: ' + city_deliver['deliver'].astype(str) + '| Delay:' + city_delay['delay'].astype(str)

data = [go.Scattergeo(
    lon = state_deliver['c_lng'],
    lat = state_deliver['c_lat'],
    text = state_deliver['text'],
    marker = dict(
        size = state_deliver['deliver']*20,
        sizemin = 1,
        color= state_delay['delay'],
        colorscale= 'Reds',
        cmin = -30,
        cmax = 0,
        line = dict(width=0.1, color='rgb(40,40,40)'),
        sizemode = 'area'
    ),
    name = 'state'),
    go.Scattergeo(
        lon = city_deliver['c_lng'],
        lat = city_deliver['c_lat'],
        text = city_deliver['text'],
        marker = dict(
            size = (city_deliver['deliver']+3),
            sizemin = 2,
            color= city_delay['delay'],
            colorscale= 'Blues',
            reversescale=True,
            cmin = -50,
            cmax = 50,
            line = dict(width=0.1, color='rgb(40,40,40)'),
            sizemode = 'area'
        ),
        name = 'city')]

layout = dict(
    title = 'Brazilian E-commerce Delivery and Delay (Click legend to toggle traces)',
    showlegend = True,
    autosize=True,
    width = 900,
    height = 600,
    geo = dict(
        scope = "south america",
        projection = dict(type='winkel tripel', scale = 1.6),
        center = dict(lon=-47,lat=-22),
        showland = True,
        showcountries= True,
        showsubunits=True,
        landcolor = 'rgb(155, 155, 155)',
        subunitwidth=1,
        countrywidth=1,
        subunitcolor="rgb(255, 255, 255)",
        countrycolor="rgb(255, 255, 255)"
    )
)
fig = dict(data=data,layout=layout)
#plot(fig,validate=False) #this is now commented so that the first plot is shown

# The North and North-Eastern regions also have higher delivery delay.