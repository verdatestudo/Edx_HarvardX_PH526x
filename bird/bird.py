'''
Week 3/4 - Case Study 5
from HarvardX: PH526x Using Python for Research on edX

In this case study, we will continue taking a look at patterns of flight for each of the three birds in our dataset.

Last Updated: 2016-Dec-24
First Created: 2016-Dec-24
Python 3.5
Chris
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

def plot_traj():
    '''
    Plot the trajectory of the three birds from the data on a 2d plane (x = long, y = lat)
    '''
    plt.figure(figsize=(7, 7))

    for bird_name in BIRD_NAMES:
        ix = BIRD_DATA.bird_name == bird_name
        x, y = BIRD_DATA.longitude[ix], BIRD_DATA.latitude[ix]
        plt.plot(x, y, '.', label=bird_name)

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(loc='lower right')
    plt.savefig('3traj.png')

def plot_speed_plt():
    '''
    Plot the speed of the birds from the provided data using plt.
    '''
    ix = BIRD_DATA.bird_name == 'Eric'
    speed = BIRD_DATA.speed_2d[ix]

    ind = np.isnan(speed) # check for nan values in speed.
    plt.hist(speed[~ind], bins=np.linspace(0, 30, 20), normed=True) # only use numeric values.
    plt.xlabel('2d speed (m/s)')
    plt.ylabel('Frequency')
    plt.savefig('eric_speed.png')

def plot_speed_pds():
    '''
    Plot the speed of the birds from the provided data using pandas.
    '''
    # one advantage of pandas is that it automatically deals with NaNs.
    BIRD_DATA.speed_2d.plot(kind='hist', range=[0, 30])
    plt.xlabel('2d speed')
    plt.show()

def convert_timestamp():
    '''
    Convert string datetime into a datetime object, and append to the BIRD_DATA.
    '''
    # second argument is the format of the string being passed.
    # capital Y = 4 digits, capital H = 24 hours.
    timestamps = [datetime.datetime.strptime(BIRD_DATA.date_time.iloc[k][:-3], '%Y-%m-%d %H:%M:%S')\
    for k in range(len(BIRD_DATA))]

    BIRD_DATA['timestamp'] = pd.Series(timestamps, index=BIRD_DATA.index)

def get_elapsed_time():
    '''
    Gets elapsed time since start of tracking.
    '''
    times = BIRD_DATA.timestamp[BIRD_DATA.bird_name == 'Eric']
    elapsed_time = [time - times[0] for time in times]
    return elapsed_time

def plot_graph_elapsed_time(elapsed_time):
    '''
    Plots a graph of elapsed time.
    '''
    plt.plot(np.array(elapsed_time) / datetime.timedelta(days=1))
    plt.xlabel('Observation')
    plt.ylabel('Elapsed time (days)')
    plt.savefig('eric_time_plot.png')


def get_daily_mean_speed(elapsed_time):
    '''
    Gets the daily mean speed of a bird and plots it on a graph.
    '''
    elapsed_days = np.array(elapsed_time) / datetime.timedelta(days=1) # convert to days.

    next_day = 1
    inds = []
    daily_mean_speed = []
    for (i, t) in enumerate(elapsed_days):
        if t < next_day:
            inds.append(i)
        else:
            daily_mean_speed.append(np.mean(BIRD_DATA.speed_2d[inds]))
            inds = []
            next_day += 1

    plt.figure(figsize=(8, 6))
    plt.plot(daily_mean_speed)
    plt.xlabel('Day')
    plt.ylabel('Mean speed (m/s)')
    plt.savefig('eric_daily_mean_speed.png')

BIRD_DATA = pd.read_csv('bird_tracking.csv')
BIRD_NAMES = pd.unique(BIRD_DATA.bird_name)

#convert_timestamp()
#elapsed_time = get_elapsed_time()
#get_daily_mean_speed(elapsed_time)

###
# pandas makes it easy to perform basic operations on groups within a dataframe without needing to loop through each value in the dataframe.
# The sample code shows you how to group the dataframe by birdname and then find the average speed_2d for each bird.
# Modify the code to assign the mean altitudes of each bird into an object called mean_altitudes.
###

# First, use `groupby` to group up the data.
grouped_birds = BIRD_DATA.groupby("bird_name")

# Now operations are performed on each group.
mean_speeds = grouped_birds.speed_2d.mean()

# The `head` method prints the first 5 lines of each bird.
# print(grouped_birds.head())

# Find the mean `altitude` for each bird.
# Assign this to `mean_altitudes`.
mean_altitudes = grouped_birds.altitude.mean()
# print(mean_altitudes)

###
###

# In this exercise, we will group the flight times by date and calculate the mean altitude within that day.
# Use groupby to group the data by date.
# Calculate the mean altitude per day and store these results as mean_altitudes_perday.

###
###

# Convert BIRD_DATA.date_time to the `pd.datetime` format.
BIRD_DATA.date_time = pd.to_datetime(BIRD_DATA.date_time)

# Create a new column of day of observation
BIRD_DATA["date"] = BIRD_DATA.date_time.dt.date

# Check the head of the column.
print(BIRD_DATA.date.head())

grouped_bydates = BIRD_DATA.groupby('date')
mean_altitudes_perday = grouped_bydates.altitude.mean()

###
###

# birddata already contains the date column.
# To find the average speed for each bird and day, create a new grouped dataframe called grouped_birdday that groups the data by both bird_name and date.

grouped_birdday = BIRD_DATA.groupby(['bird_name', 'date'], as_index=False)
mean_altitudes_perday = grouped_birdday.altitude.mean()

# look at the head of `mean_altitudes_perday`.
print(mean_altitudes_perday.head())

###
###

# Great! Now find the average speed for each bird and day. Create three dataframes – one for each bird.
# Use the plotting code provided to plot the average speeds for each bird.

mean_speeds_day = grouped_birdday.speed_2d.mean()

eric_daily_speed  = mean_speeds_day[mean_speeds_day.bird_name =='Eric']
sanne_daily_speed = mean_speeds_day[mean_speeds_day.bird_name =='Sanne']
nico_daily_speed  = mean_speeds_day[mean_speeds_day.bird_name =='Nico']

plt.plot(eric_daily_speed.date, eric_daily_speed.speed_2d, label="Eric")
plt.plot(sanne_daily_speed.date, sanne_daily_speed.speed_2d, label="Sanne")
plt.plot(nico_daily_speed.date, nico_daily_speed.speed_2d, label="Nico")
plt.legend(loc="upper left")
plt.show()

###
###
