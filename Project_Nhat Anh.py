#!/usr/bin/env python
# coding: utf-8

# # TABLE OF CONTENT

# * Data cleaning
# 
# * EDA
# 
# 1. PRICE
# 
#     1.1.	Comparing price of each room type between Resort hotel and City hotel
#     
#     1.2.	Which month have the highest room price?
#     
#     1.3.	Which is the busiest month?
#     
#     1.4.	Where most income come from?
#     
# * Yearly adr by country
# * Yearly adr customer 
# 
# 
# 2. GUEST
# 
#     2.1.	From where the most guests are coming ?
#     
#     2.2.	How long does people stay in the hotel?
#     
# 
# 3. MARKET SEGMENT 
# 
#     3.1. Number of Guests by Market Segment
#     
#     3.2. Distribution of Market Segment by different Hotel type
#     
# 
# 4. CANCELLATION RATE
# 
#     4.1. Changing in number of cancelled booking in a period of 3 years.
#     
#     4.2. Number of cancellation by room type
#     
#     4.3. Number of cancellation and total_or_special_request
#     
#     4.4. Number of booking cancellation in Resort hotel and City hotel
#     
#     4.5. Distribution of number of booking cancellation and lead time
#         
#     4.6. Number of booking and number of cancellation in each month of year
#     
#     4.7. Number of booking and number of cancellation by distribution channel
# 

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium


# In[4]:


import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns',None)


# # Data Cleaning

# In[5]:


df= pd.read_csv('hotel_bookings.csv')


# In[6]:


df.head()


# In[7]:


df.isnull().sum()


# * Missing values: children, country, agent, company
# * Country is the only categorical data that contain missing values others are numerical
# It can be seen that there are a total of four columns with missing values, which are handled as follows:
# 
# <!-- The children and country columns are filled with the mode of the corresponding column due to the small number of missing;
# 
# The agent is missing 16,340, the missing rate is 13.6%, and the number of missing is large, but the agent indicates the booked travel agency, and the missing rate is less than 20%, keep it and fill it with 0, which means there is no travel agency ID;
# 
# The company is missing 112593, the missing rate is 94.3%>80%, it does not have the validity of the information value, so it is deleted directly -->

# In[8]:


### replacing null data in agent column
median = df.agent.median()
df.agent.fillna(median,inplace=True)


# In[9]:


### replacing null data in children column
median = df.children.median()
df.children.fillna(median,inplace=True)


# In[10]:


### replacing null data in country column
df.country.isnull().sum()/df.country.count() #### as the data is less than 40 % we can use the mode to impute the NAN values


# In[11]:


mode = df.country.mode()[0]
df.country.fillna(mode,inplace=True)


# In[12]:


### dropping the column company
df.drop('company',axis=1,inplace=True)


# In[13]:


df.isnull().sum()


# In[14]:


filter = (df.adults ==0) & (df.children ==0) & (df.babies ==0) 
df = df[~filter]


# Changing arrival year, month and day feature to datetime format

# In[15]:


df['arrival_date_year'] = pd.to_datetime(df.arrival_date_year,format= '%Y' ).dt.year
df['arrival_date_month'] = pd.to_datetime(df.arrival_date_month,format= '%B' ).dt.month
df['arrival_date_day_of_month'] = pd.to_datetime(df.arrival_date_day_of_month,format= '%d' ).dt.day


# To calculate the actual number of orders, it is necessary to remove those cancelled orders

# In[16]:


df_not_cancelled = df[df['is_canceled']==0]
df_cancelled = df[df['is_canceled']==1]


# # EDA

# ## 1. Price

# ### Comparing price of each room type between Resort hotel and City hotel

# In[17]:


fig = px.box(data_frame=df_not_cancelled, x='reserved_room_type', y='adr', color='hotel')
fig.update_layout(title_text='Distribution of Room Type and ADR',
                  title_x=0.5, title_font=dict(size=20))
fig.update_traces(marker=dict(line=dict(width=1)))
fig.show()


# * Resort hotel generally has lower adr (average daily rate) compared to that of City hotel for all room type except for type room C. 
# * However, City hotel' s adr has higher variance.

# ### 1.1. Which month have the highest room price?

# In[18]:


stay = df_not_cancelled.groupby(['arrival_date_month','hotel']).agg({'adr':'mean'}).reset_index()
stay.head()


# In[19]:


px.line(data_frame=stay, 
        x= "arrival_date_month", 
        y= "adr", 
        line_group= "hotel", 
        color="hotel", 
        title= "Which month have the highest room price?",
        labels = {"arrival_date_month": "Month", "adr": "Monthly adr"}
       )


# * For resort hotel, room price was highest in August and dropped rapidly to the lowest point in November before increasing again in the next 9 months. 
# 
# * Meanwhile, City hotel price was more stable throughout the year and reached the highest value in the period from May to August. 

# ### 1.2. Which is the busiest month?

# In[20]:


df_not_cancelled ['total guest'] = df_not_cancelled ['adults'] + df_not_cancelled ['children'] + df_not_cancelled ['babies']


# In[21]:


stay = df_not_cancelled.groupby(['arrival_date_month','hotel']).agg({'total guest':'sum'}).reset_index()
stay


# In[22]:


px.line(data_frame=stay, 
        x= "arrival_date_month", 
        y= "total guest", 
        line_group= "hotel", 
        color="hotel", 
        title= "Which is the busiest month?",
        labels = {"arrival_date_month": "Month", "total guest": "Num of guest"}
       )


# August is the busiest time in the year for both resort and city hotel. From graph 1.1 and 1.2, we can see that August is the busiest month and has the highest adr. The correlation makes sense because the supply of traditional hotel rooms is pretty much fixed in the short term: when demand increases in the peak season, you can’t just instantly build more hotels to meet the opportunity. That means the price for a hotel room skyrockets for travelers during the busy season. However, if hotels raise their prices too much, they risk a pretty bad outcome: the rooms could sit empty and not generate any revenue at all!
# 
# Hotels play a delicate balancing act with their pricing, aided by revenue management software. They need to predict when demand will be high, and just how richly they can price the room based on competition, local events, and people’s vacation schedules. At the same time, during periods of lower demand, they need to make sure the prices are enticing enough that their rooms aren’t sitting empty. 

# ### 1.3. Where most income come from?

# #### a) Yearly adr by country

# ##### City Hotel

# In[63]:


data_City= df_not_cancelled[(df_not_cancelled['hotel'] == 'City Hotel')]
top_10_country_with_highest_yearly_adr_City_hotel= data_City.groupby(['country']).agg({'adr':'mean'}).nlargest(10, 'adr') 


# In[24]:


#list of 10 country that pay the most yearly adr to City Hotel
top_10_country = []
top_10_country.extend(top_10_country_with_highest_yearly_adr_City_hotel.index.tolist())
top_10_country


# In[64]:


data= data_City.loc[data_City['country'].isin(top_10_country)]
yearly_adr_of_top_10_country = data.groupby(['arrival_date_year', 'country']).agg({'adr':'mean'}).reset_index()
# yearly_adr_of_top_10_country['year']= yearly_adr_of_top_10_country.index()


# In[65]:


yearly_adr_of_top_10_country.head()


# In[66]:


px.line(data_frame=yearly_adr_of_top_10_country, 
        x= 'arrival_date_year', 
        y= "adr", 
        line_group= "country", 
        color="country", 
        title= "Top 10 country paying highest yearly adr to City Hotel",
        labels = {"arrival_date_year": "Year", "adr": "Income"}
       )


# * CHE, USA and ITA is the top 3 country from which the customer paid for the highest room.
# * There needs to be more focus on these type of customer to utilize their experience and encourage them to comeback. 

# ##### Resort Hotel

# In[68]:


data_Resort= df_not_cancelled[(df_not_cancelled['hotel'] == 'Resort Hotel')]
top_10_country_with_highest_yearly_adr_Resort_hotel= data_Resort.groupby(['country']).agg({'adr':'mean'}).nlargest(10, 'adr') 

#list of 10 country that pay the most yearly adr to City Hotel
top_10_country = []
top_10_country.extend(top_10_country_with_highest_yearly_adr_Resort_hotel.index.tolist())

data= data_Resort.loc[data_Resort['country'].isin(top_10_country)]
yearly_adr_of_top_10_country = data.groupby(['arrival_date_year', 'country']).agg({'adr':'mean'}).reset_index()
# yearly_adr_of_top_10_country['year']= yearly_adr_of_top_10_country.index()

yearly_adr_of_top_10_country
px.line(data_frame=yearly_adr_of_top_10_country, 
        x= 'arrival_date_year', 
        y= "adr", 
        line_group= "country", 
        color="country", 
        title= "Top 10 country paying highest yearly adr to Resort Hotel",
        labels = {"arrival_date_year": "Year", "adr": "yearly adr"}
       )


# * Customer booking Resort hotel mostly came from 4 country: AND, AZE, VNM and PHL. AND is the country that has most customer booking these resort hotel. 
# * Adr increases gradually each year, AND's adr rose most considerally.

# ### b) Yearly adr by customer type

# In[29]:


df_customer_type = df_not_cancelled.groupby(['customer_type','arrival_date_year']).agg({'adr':'sum', 'total guest':'sum'}).reset_index()


# In[67]:


trace1 = go.Bar(x= data[(data["arrival_date_year"] == 2015)]["customer_type"], 
                y= data[(data["arrival_date_year"] == 2015)]["adr"], 
                name= "2015")
trace2 = go.Bar(x= data[(data["arrival_date_year"] == 2016)]["customer_type"], 
                y= data[(data["arrival_date_year"] == 2016)]["adr"], 
                name= "2016")
trace3 = go.Bar(x= data[(data["arrival_date_year"] == 2017)]["customer_type"], 
                y= data[(data["arrival_date_year"] == 2017)]["adr"], 
                name= "2017")
X= [trace1, trace2, trace3]
figure = go.Figure(data= X)
figure.show()


# ## 2. GUEST

# ### 2.1. From where the most guests are coming ?

# In[31]:


country_guests = df_not_cancelled['country'].value_counts().reset_index()
country_guests.columns = ['country', 'No of guests']
country_guests


# In[32]:


basemap = folium.Map()
guests_map = px.choropleth(country_guests, 
                           locations = country_guests['country'],
                           color = country_guests['No of guests'], 
                           hover_name = country_guests['country'])
guests_map.show()


# Most gueseses come from Europe, especially Portugal.

# ### 2.2. How long does people stay in the hotel?

# In[33]:


df_not_cancelled ['total_nights'] = df_not_cancelled ['stays_in_weekend_nights'] + df_not_cancelled ['stays_in_week_nights']
stay = df_not_cancelled.groupby(['total_nights', 'hotel']).agg('count').reset_index()
stay = stay.iloc[:, :3]
stay = stay.rename(columns={'is_canceled':'Number of stays'}) 
#chọn is_canceled vì chắc chắn cột này cho số liệu bằng với số booking thỏa điều kiện tương ứng (lưu ý đang xét trong data là not_cancelled)


# In[34]:


px.bar(data_frame = stay, 
       x = 'total_nights', 
       y = 'Number of stays', 
       color = 'hotel', 
       barmode = 'group')


# Most people stay in city hotel for 3 days while stay in resort hotel for 1 day.

# # 3. Market Segment

# ### 3.1. Number of Guests by Market Segment

# In[35]:


df_not_cancelled['arrival_date']= pd.to_datetime(dict(year=df['arrival_date_year'], 
                                       month=df['arrival_date_month'], 
                                       day=df['arrival_date_day_of_month']))


# In[36]:


import plotly.subplots as sp


# In[37]:


df_market_segment = df_not_cancelled.groupby(['market_segment','arrival_date_year', 'arrival_date_month']).agg({'total guest':'sum'}).reset_index()
df_market_segment_2015 = df_market_segment [df_market_segment['arrival_date_year'] == 2015]
df_market_segment_2016 = df_market_segment [df_market_segment['arrival_date_year'] == 2016]
df_market_segment_2017 = df_market_segment [df_market_segment['arrival_date_year'] == 2017]


# In[38]:


figure1 = px.line(data_frame=df_market_segment_2015, 
        x= "arrival_date_month", 
        y= "total guest", 
        line_group= "market_segment", 
        color="market_segment", 
        title= "Number of Guests by Market Segment in 2015",
        labels = {"arrival_date": "Time", "total guest": "Total guest"}
       )
figure2 = px.line(data_frame=df_market_segment_2016, 
        x= "arrival_date_month", 
        y= "total guest", 
        line_group= "market_segment", 
        color="market_segment", 
        title= "Number of Guests by Market Segment in 2015",
        labels = {"arrival_date": "Time", "total guest": "Total guest"}
       )
figure3 = px.line(data_frame=df_market_segment_2017, 
        x= "arrival_date_month", 
        y= "total guest", 
        line_group= "market_segment", 
        color="market_segment", 
        title= "Number of Guests by Market Segment in 2015",
        labels = {"arrival_date": "Time", "total guest": "Total guest"}
       )


figure1_traces = []
figure2_traces = []
figure3_traces = []
for trace in range(len(figure1["data"])):
    figure1_traces.append(figure1["data"][trace])
for trace in range(len(figure2["data"])):
    figure2_traces.append(figure2["data"][trace])
for trace in range(len(figure3["data"])):
    figure3_traces.append(figure3["data"][trace])
    
    
    
#Create a 1x2 subplot
this_figure = sp.make_subplots(rows=3, cols=1) 

# Get the Express fig broken down as traces and add the traces to the proper plot within in the subplot
for traces in figure1_traces:
    this_figure.append_trace(traces, row=1, col=1)
for traces in figure2_traces:
    this_figure.append_trace(traces, row=2, col=1)
for traces in figure3_traces:
    this_figure.append_trace(traces, row=3, col=1)
this_figure.show()


# ### 3.2. Distribution of Market Segment by different Hotel type

# In[39]:


Market_segment_fig = px.histogram(df, x="market_segment", color='hotel')
Market_segment_fig.update_layout(barmode='group', xaxis={'categoryorder': 'total descending'})
Market_segment_fig.update_traces(marker=dict(line=dict(color='#000000', width=1)))
Market_segment_fig.update_layout(title_text='Distribution of Market Segment by Resort Hotel and City Hotel',
                  title_x=0.5, title_font=dict(size=20))
Market_segment_fig.show()


# # 4. Cancellation

# ### 4.1. Changing in number of cancelled booking in a period of 3 years.

# In[40]:


df_cancelled.arrival_date_year.value_counts().plot.bar(figsize=(6,6))


# ### 4.2. Number of cancellation by room type

# In[41]:


sns.countplot(x=df['customer_type'], hue=df['is_canceled'])


# The customer type is mainly past travelers and the order cancellation rate is high
# 
# Group type customers have the least but almost no cancellations (I don’t understand the difference between Transient-Party and Group here. I understand that Transient-Party is a travel agency, and Group is a company group. If this is the case, it’s not difficult to understand. Company group building is generally It is customary and will not be easily cancelled due to personal reasons, while travel agencies are mainly dependent on personal wishes, so the probability of cancellation is higher)
# 
# On the whole, group cancellation probability is lower than in dividual cancellation probability. Group purchase discounts can be set to ensure the stability of passenger flow.

# ### 4.3. Number of cancellation and total_or_special_request

# In[42]:


sns.countplot(x=df['total_of_special_requests'],hue=df['is_canceled'])


# Most customers have no special needs
# 
# The results did not show that there were many special needs of customers and the high cancellation rate, but the opposite trend, indicating that the hotel has certain characteristics and has done a good job in meeting the special needs of customers and continues to maintain it.

# ### 4.4. Number of booking cancellation in Resort hotel and City hotel

# In[69]:


total_cancelations = df["is_canceled"].sum()
resort_hotel_cancelations = df.loc[df["hotel"] == "Resort Hotel"]["is_canceled"].sum()
city_hotel_cancelations = df.loc[df["hotel"] == "City Hotel"]["is_canceled"].sum()

rel_cancel = total_cancelations / df.shape[0] * 100
rh_rel_cancel = resort_hotel_cancelations / df.loc[df["hotel"] == "Resort Hotel"].shape[0] * 100
ch_rel_cancel = city_hotel_cancelations / df.loc[df["hotel"] == "City Hotel"].shape[0] * 100

print(f"Total bookings canceled: {total_cancelations:,} ({rel_cancel:.0f} %)")
print(f"Resort hotel bookings canceled: {resort_hotel_cancelations:,} ({rh_rel_cancel:.0f} %)")
print(f"City hotel bookings canceled: {city_hotel_cancelations:,} ({ch_rel_cancel:.0f} %)")


# The cancellation rate of city hotel reservations is higher than that of resort hotels, mainly because the main user group of city hotels is business travel users, which are often urgent and unplanned. Hotel reservations are easy without planning and in-depth understanding of the state of the hotel. Blindly book and unsubscribe, so the unsubscribe rate is high. It is recommended to add the "Nearby Preferred" function to the channel platform, and automatically filter and recommend nearby cities with high occupancy rates, high re-occupation rates, high evaluations and other high-quality feedback by entering addresses. City hotels, on the one hand, can provide users with more efficient and convenient recommendation services, and on the other hand, they can also promote platform channels to optimize service content.
# 

# ### 4.5. Distribution of number of booking cancellation and lead time

# In[44]:


plt.plot(df.groupby('lead_time')['is_canceled'].mean().index,
         df.groupby('lead_time')['is_canceled'].mean(),
         'ro',
         markersize=2)


# * The smaller the lead time is, the lower the cancellation probability. 
# * As the booking advance time increases, the cancellation rate presents an upward trend and in a more unpredictable way.

# ### 4.6. Number of booking and number of cancellation in each month of year by City Hotel and Resort Hotel

# In[83]:


ordered_months = range(1, 13)
for hotel in ['City Hotel','Resort Hotel']:
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    df_hotel = df[(df['hotel']==hotel) & (df['arrival_date_year']==2016) ]
    monthly_order = df_hotel.groupby('arrival_date_month').size()
    # barplot
    sns.barplot(x=list(range(1, 13)),
                y= monthly_order[ordered_months], 
                ax=ax1)
    # plot
    ax2.plot(range(12), 
             df_hotel.groupby('arrival_date_month')['is_canceled'].sum()[ordered_months].values, 
             'ro-')
    
    ax1.set_xlabel('Month')
    ax2.set_ylabel('Rate of cancellation')
    ax1.set_ylabel('Number of booking')
    plt.title(hotel)


# * In generall, rate of cancellation fluctuated in the same trend with number of booking throughout 2015.
# * October always had the highest cancellation rate in both resort hotel and city hotel. From January to April, both type of hotel witnessed an increasing number of booking as well as rate of cancellation. In the same fashion, both type of hotel had a decreasing period of cancellation rate and number of booling.
# * January is the month that had lowest rate of cancellation as well as number of booking. 

# ### 4.7. Number of booking and number of cancellation by distribution channel

# ##### City Hotel

# In[84]:


df_hotel=df[(df['hotel']=='City Hotel') & (df['arrival_date_year']==2016)]
distribution_channel_unique = df_hotel['distribution_channel'].unique().tolist()

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
    
num_of_booking_by_distribution_channel=df_hotel.groupby('distribution_channel').size()
    
# barplot
sns.barplot(x=distribution_channel_unique ,
                y= num_of_booking_by_distribution_channel[distribution_channel_unique], 
                ax=ax1)
    
# plot
ax2.plot(df_hotel.groupby('distribution_channel')['is_canceled'].sum(), 
             'ro-')
    
 # Set title, label
ax1.set_ylabel('Number of booking')
ax2.set_ylabel('Number of cancellation')
ax1.set_xlabel('Distribution Channel')
plt.title("City Hotel")


# ##### Resort Hotel

# In[86]:


df_hotel=df[(df['hotel']=='Resort Hotel') & (df['arrival_date_year']==2016)]
distribution_channel_unique = df_hotel['distribution_channel'].unique().tolist()
  

fig, ax1 = plt.subplots()
ax2 = ax1.twinx() 

num_of_booking_by_distribution_channel=df_hotel.groupby('distribution_channel').size()

# barplot
sns.barplot(x=distribution_channel_unique ,
            y= num_of_booking_by_distribution_channel[distribution_channel_unique], 
            ax=ax1)
    
# plot
ax2.plot(df_hotel.groupby('distribution_channel')['is_canceled'].sum(), 
            'ro-')
 # Set title, label
ax1.set_ylabel('Number of booking')
ax2.set_ylabel('Number of cancellation')
ax1.set_xlabel('Distribution Channel')
plt.title("Resort Hotel")


# 1) Reservations are mainly from direct channel and corporate channel.
# 
# 2) The probability of a travel agency canceling a reservation is much greater than that of other channels. It may be because travel agencies will cancel lower-profit orders out of profit considerations.
# 
# The almost perfect convenience of an instant and comfortable online booking process, along with carefree “Click to Cancel” policies have encouraged this flip-flop behavior. Guests are able to secure their stay at a hotel one moment, and cancel the next moment, sometimes after finding a better last-minute deal at another hotel. Obviously, providing a better experience for the guest, including easy booking and cancellations, is a win-win for the travel and hospitality industry.
# 
# The reports states that guests have become accustomed to free cancellation policies that have been made popular (and encouraged) mainly by online channels and apps, which are designed to cancel and rebook hotel rooms at each rate drop.
# 
# That behavioral shift in booking has not only increased hotel cancellation rates, but has dramatically increased revenue losses for hotels. Hoteliers are now facing a big obstacle for revenue management, inventory distribution and rates optimization. As you can see in the graphs above, OTAs contribute most to this problem, while cancellations from direct booking guests remain low.

# In[ ]:




