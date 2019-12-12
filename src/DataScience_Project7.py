# # The Oscars

# Last summer, I worked through the University of Rhode Island and collected data for a local oyster farm using different instruments like tilt current meters and a water level logger. The tilt current meter was used to measure the current's speed and direction and water temperature. The water level logger was used to measure the water depth and temperature. The pond, Potter Pond, that houses the oyster farm, Matunuck Oyster Farm, has very high spatial variablility. This makes it very to difficult to find patterns within different phsycial conditions like temperature, current speed, and current direction. It was my job to vizualize this data and attempt to uncover these patterns using MATLAB. I understand that this is not data used in a previous project. However, I wanted to see if I could manipulate this data with Python/ Pandas and use the knowlege from this class to get better vizualizations.


import matplotlib.pyplot as plt
import pandas as pd
import datetime
import time
import numpy as np
from scipy.signal import medfilt
import matplotlib.dates as mdates

#adds datetime to data
def addTime(fname):
    df=pd.read_csv(fname, skiprows=[0])
    timeList=df['time'].tolist()
    dateList=[datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%f') for x in timeList]
    utime=[time.mktime(y.timetuple()) for y in dateList]
    df.insert(1,'datetime',dateList,True)
    df.insert(2,'unixtime',utime,True)
    return(df)

#adds datetime to water level logger data
def addDepthTime(pname):
    df=pd.read_csv(pname, skiprows=[0])
    timeList = df['time'].tolist()
    dateList=[datetime.datetime.strptime(x, '%m-%d-%YT%H:%M:%S') for x in timeList]
    utime=[time.mktime(y.timetuple()) for y in dateList]
    df.insert(1,'datetime',dateList,True)
    df.insert(2,'unixtime',utime,True)
    return(df)

apriltemp_df=addTime('../data/1806202_logger_04172019-06042019_T.txt')
aprilcurrent_df=addTime('../data/1806202_logger_04172019-06042019_CR.txt')

julytemp_df=addTime('../data/1806202_logger_07102019-07172019_T.txt')
julycurrent_df=addTime('../data/1806202_logger_07102019-07172019_CR.txt')
julydepth_df=addDepthTime('../data/190710-190717_depth_SN416.txt')

junetemp_df=addTime('../data/1806202_logger_06042019-07022019_T.txt')
junecurrent_df=addTime('../data/1806202_logger_06042019-07022019_CR.txt')
junedepth_df=addDepthTime('../data/190604-190702_depth_SN416.txt')

juneLongtemp_df=addTime('../data/1805228_logger_06042019-07022019_T.txt')
juneLongcurrent_df=addTime('../data/1805228_logger_06042019-07022019_CR.txt')

print(aprilcurrent_df.head())
print(junecurrent_df.head())
print(julycurrent_df.head())
print(apriltemp_df.head())
print(junetemp_df.head())
print(julytemp_df.head())
print(junedepth_df.head())
print(julydepth_df.head())

#This function is used to calculate the time of rising and falling tide
def getRising(df_depth):
    m=medfilt(np.diff(df_depth['pressure']),9)
    rising=np.where(m>=0)[0]
    return(rising)
#gets indices of falling tide
def getFalling(df_depth):
    m=medfilt(np.diff(df_depth['pressure']),9)
    falling=np.where(m<0)[0]
    return(falling)

# function used that I converted from MATLAB to Python that makes different plots
def make_plots(df_current, df_temp, df_depth):
    temp_interp=np.interp(df_temp['unixtime'],df_current['unixtime'],df_temp['temp'])
    fig2=plt.figure()
    ax=fig2.add_subplot(111)
    ax.plot(df_current['datetime'],df_current['speed'],'.',markersize=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y %H:%M:%S'))
    plt.xlabel("Date")
    plt.ylabel("Current Speed (cm/s)")
    plt.title("Current Speed Scatter Plot")
    fig2.autofmt_xdate()
    plt.grid()
    plt.show()
    fig3=plt.figure()
    ax=fig3.add_subplot(111)
    ax.plot(df_current['datetime'],df_current['bearing'],'.',markersize=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y %H:%M:%S'))
    plt.xlabel("Date")
    plt.ylabel("Current Bearing (degrees going towards)")
    plt.title("Current Bearing Scatter Plot")
    fig3.autofmt_xdate()
    plt.grid()
    plt.show()
    fig4=plt.figure()
    ax=fig4.add_subplot(111)
    plt.plot(df_temp['datetime'],df_temp['temp'],'.',markersize=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y %H:%M:%S'))
    plt.xlabel("Date")
    plt.ylabel("Temperature (C)")
    plt.title("Temperature Scatter Plot")
    fig4.autofmt_xdate()
    plt.grid()
    plt.show()
    fig5=plt.figure()
    ax=fig5.add_subplot(111)
    plt.plot(df_depth['datetime'],df_depth['pressure'],'.',markersize=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y %H:%M:%S'))
    plt.xlabel("Date")
    plt.ylabel("Depth Pressure (psi)")
    plt.title("Depth Pressure Scatter Plot")
    fig5.autofmt_xdate()
    plt.grid()
    plt.show()
    fig6=plt.figure()
    ax=fig6.add_subplot(111)
    plt.plot(df_depth['datetime'],df_depth['temp'],'.',markersize=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y %H:%M:%S'))
    plt.xlabel("Date")
    plt.ylabel("Depth Temperature (C)")
    plt.title("Depth Temperature Scatter Plot")
    fig6.autofmt_xdate()
    plt.grid()
    plt.show()
    plt.figure()
    plt.plot(df_current['bearing'],temp_interp,'.',markersize=2)
    plt.xlabel("Current Bearing (deg going towards)")
    plt.ylabel("Temperature (C)")
    plt.title("Temperature vs Current Bearing Scatter Plot")
    plt.grid()
    plt.show()
    plt.figure()
    plt.plot(df_current['speed'],temp_interp,'.',markersize=2)
    plt.xlabel("Current Speed (cm/s)")
    plt.ylabel("Temperature (C)")
    plt.title("Temperature vs Current Speed Scatter Plot")
    plt.grid()
    plt.show()

make_plots(junecurrent_df,junetemp_df,junedepth_df)

make_plots(julycurrent_df,julytemp_df,julydepth_df)
# calculates when rising and falling tides occur
def getRF(df_current,df_depth):
    rising=getRising(df_depth)
    falling=getFalling(df_depth)
    current_rising_interp=np.interp(df_depth['unixtime'][rising],df_current['unixtime'],df_current['speed'])
    current_falling_interp=np.interp(df_depth['unixtime'][falling],df_current['unixtime'],df_current['speed'])
    bearing_rising_interp=np.interp(df_depth['unixtime'][rising],df_current['unixtime'],df_current['bearing'])
    bearing_falling_interp=np.interp(df_depth['unixtime'][falling],df_current['unixtime'],df_current['bearing'])

    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    ax.plot(df_depth['datetime'][rising],df_depth['pressure'][rising],'r.', markersize=2, label='Rising Tide')
    ax.plot(df_depth['datetime'][falling],df_depth['pressure'][falling], 'b.',markersize=2, label='Falling Tide')
    # format your data to desired format. Here I chose YYYY-MM-DD but you can set it to whatever you want.
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y %H:%M:%S'))
    # rotate and align the tick labels so they look better
    fig1.autofmt_xdate()
    plt.xlabel("Date")
    plt.title("Rising and Falling Tides Scatter Plot")
    plt.ylabel("Depth Pressure (psi)")
    legend = plt.legend(loc='lower right')
    legend.legendHandles[0]._sizes = [10]
    legend.legendHandles[1]._sizes = [20]
    plt.grid()
    plt.show()
getRF(junecurrent_df,junedepth_df)
getRF(julycurrent_df,julydepth_df)

def make_hist(df_current,df_depth):
    rising=getRising(df_depth)
    falling=getFalling(df_depth)
    current_rising_interp=np.interp(df_depth['unixtime'][rising],df_current['unixtime'],df_current['speed'])
    current_falling_interp=np.interp(df_depth['unixtime'][falling],df_current['unixtime'],df_current['speed'])
    bearing_rising_interp=np.interp(df_depth['unixtime'][rising],df_current['unixtime'],df_current['bearing'])
    bearing_falling_interp=np.interp(df_depth['unixtime'][falling],df_current['unixtime'],df_current['bearing'])

    fig4 = plt.figure()
    ax = fig4.add_subplot(111)
    plt.hist2d(bearing_rising_interp,df_depth['pressure'][rising], bins=(50,20))
    plt.axis([0, 360,15.6, 16.4])
    plt.xlabel("Current Bearing (deg going towards)")
    plt.ylabel("Depth Pressure (psi)")
    plt.title("Rising Tide Depth Pressure vs Current Bearing Histogram")
    plt.colorbar()
    plt.grid()
    plt.show()

    fig5 = plt.figure()
    ax = fig5.add_subplot(111)
    plt.hist2d(bearing_falling_interp,df_depth['pressure'][falling], bins=(50,20))
    plt.axis([0, 360,15.6, 16.4])
    plt.xlabel("Current Bearing (deg going towards)")
    plt.ylabel("Depth Pressure (psi)")
    plt.title("Falling Tide Depth Pressure vs Current Bearing Histogram")
    plt.colorbar()
    plt.grid()
    plt.show()
make_hist(junecurrent_df,junedepth_df)

make_hist(julycurrent_df,julydepth_df)
def checkSpeed(df_current,df_depth):
    rising=getRising(df_depth)
    falling=getFalling(df_depth)
    current_rising_interp=np.interp(df_depth['unixtime'][rising],df_current['unixtime'],df_current['speed'])
    current_falling_interp=np.interp(df_depth['unixtime'][falling],df_current['unixtime'],df_current['speed'])
    bearing_rising_interp=np.interp(df_depth['unixtime'][rising],df_current['unixtime'],df_current['bearing'])
    bearing_falling_interp=np.interp(df_depth['unixtime'][falling],df_current['unixtime'],df_current['bearing'])

    fig2 = plt.figure()
    plt.hist2d(current_rising_interp,df_depth['pressure'][rising], bins=(250,50))
    plt.axis([0, 4,15.6, 16.4])
    plt.xlabel("Current Speed (cm/s)")
    plt.ylabel("Depth Pressure (psi)")
    plt.title("Rising Tide Depth Pressure vs Current Speed Histogram")
    plt.colorbar()
    plt.grid()
    plt.show()

    fig3 = plt.figure()
    plt.hist2d(current_falling_interp,df_depth['pressure'][falling], bins=(250,50))
    plt.axis([0, 4,15.6, 16.4])
    plt.xlabel("Current Speed (cm/s)")
    plt.ylabel("Depth Pressure (psi)")
    plt.title("Falling Tide Depth Pressure vs Current Speed Histogram")
    plt.colorbar()
    plt.grid()
    plt.show()

checkSpeed(junecurrent_df,junedepth_df)
checkSpeed(julycurrent_df,julydepth_df)

apriltemp_df['day'] = pd.DatetimeIndex(apriltemp_df['datetime']).day
apriltemp_df['month'] = pd.DatetimeIndex(apriltemp_df['datetime']).month
apriltemp_df['year'] = pd.DatetimeIndex(apriltemp_df['datetime']).year

junetemp_df['day'] = pd.DatetimeIndex(junetemp_df['datetime']).day
junetemp_df['month'] = pd.DatetimeIndex(junetemp_df['datetime']).month
junetemp_df['year'] = pd.DatetimeIndex(junetemp_df['datetime']).year

julytemp_df['day'] = pd.DatetimeIndex(julytemp_df['datetime']).day
julytemp_df['month'] = pd.DatetimeIndex(julytemp_df['datetime']).month
julytemp_df['year'] = pd.DatetimeIndex(julytemp_df['datetime']).year
result=pd.concat([apriltemp_df, junetemp_df, julytemp_df])
print(result.head(10))

result['monthday'] = result['datetime'].apply(lambda x:x.strftime('%m/%d'))
result.dtypes
import seaborn as sns
fig = plt.subplots(1, figsize=(20,7), dpi= 80)
plt.xticks(rotation=90)
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
country_firstleague_ranks =sns.boxplot(x='monthday', y='temp', data=result).set(
    xlabel='Month/ Day',
    ylabel='Temperature'
)


import pandas as pd
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource
from bokeh.palettes import Spectral3
from bokeh.models.annotations import Title
output_file('temp_lineplot.html')

source1 = ColumnDataSource(apriltemp_df)
source2 = ColumnDataSource(junetemp_df)
source3 = ColumnDataSource(julytemp_df)

p = figure(x_axis_type='datetime')
t=Title()
t.text = 'Change in Temperature'
p.title = t
p.line(x='datetime', y='temp', line_width=2, source=source1, legend='April-June Temp', color='red')
p.line(x='datetime', y='temp', line_width=2, source=source2, legend='June-July Temp', color='blue')
p.line(x='datetime', y='temp', line_width=2, source=source3, legend='July Temp', color='green')
p.yaxis.axis_label = 'Temperature'
show(p)

df1=junetemp_df.groupby(junetemp_df.datetime.dt.date).mean()
df2=junecurrent_df.groupby(junecurrent_df.datetime.dt.date).mean()

def fahr_to_celsius(temp_fahr):
    temp_celsius = (temp_fahr - 32) * 5.0 / 9.0
    return temp_celsius

junedepth_df['temp']=fahr_to_celsius(junedepth_df['temp'])

df3=junedepth_df.groupby(junecurrent_df.datetime.dt.date).mean()

result2 = pd.merge(df1, df2, on='datetime')
result2 = pd.merge(result2, df3, on='datetime')
result2.head()
fig, ax = plt.subplots(figsize=(10,10))
cl = result2[['temp_x','temp_y','speed','bearing', 'pressure']].corr()
sns.heatmap(cl, square = True, ax=ax)

rising=getRising(junedepth_df)
falling=getFalling(junedepth_df)
LR=list(juneLongcurrent_df['bearing'][rising])
LF=list(juneLongcurrent_df['bearing'][falling])
SR=list(junecurrent_df['bearing'][rising])
SF=list(junecurrent_df['bearing'][falling])

f={'LongFalling': LF, 'ShortFalling': SF}
r={'LongRising': LR, 'ShortRising': SR}
df_JuneFalling = pd.DataFrame(f)
df_JuneRising=pd.DataFrame(r)

N = 20
bottom = 8
max_height = 4

radii = juneLongcurrent_df['speed']
theta = juneLongcurrent_df['bearing']
width = (2*np.pi) / N

ax = plt.subplot(111, polar=True)
bars = ax.bar(theta, radii, width=width, bottom=bottom)

# Use custom colors and opacity
for r, bar in zip(radii, bars):
    bar.set_facecolor(plt.cm.jet(r / 10.))
    bar.set_alpha(0.8)

plt.show()

N = 20
bottom = 8
max_height = 4

radii = junecurrent_df['speed']
theta = junecurrent_df['bearing']
width = (2*np.pi) / N

ax = plt.subplot(111, polar=True)
bars = ax.bar(theta, radii, width=width, bottom=bottom)

# Use custom colors and opacity
for r, bar in zip(radii, bars):
    bar.set_facecolor(plt.cm.jet(r / 10.))
    bar.set_alpha(0.8)

plt.show()

output_file('fallingCorrelation.html')

source1 = ColumnDataSource(df_JuneFalling)
source2 = ColumnDataSource(df_JuneRising)
x=df_JuneFalling['ShortFalling']
y=df_JuneFalling['LongFalling']

# determine best fit line
par = np.polyfit(x, y, 1, full=True)
slope=par[0][0]
intercept=par[0][1]
y_predicted = [slope*i + intercept  for i in x]

# plot it
fig=figure()
t=Title()
t.text = 'Wind Direction vs Current Direction During Falling Tide'
fig.title = t
fig.circle(x,y, legend='Data')
fig.line(x,y_predicted,color='red',legend= "Regression Line")
fig.yaxis.axis_label = 'Wind Direction During Falling Tide (deg)'
fig.xaxis.axis_label = 'Current Direction During Falling Tide (deg)'
show(fig)


output_file('risingCorrelation.html')

x=df_JuneRising['ShortRising']
y=df_JuneRising['LongRising']

# determine best fit line
par = np.polyfit(x, y, 1, full=True)
slope=par[0][0]
intercept=par[0][1]
y_predicted = [slope*i + intercept  for i in x]

# plot it
fig=figure()
t=Title()
t.text = 'Wind Direction vs Current Direction During Rising Tide'
fig.title = t
fig.circle(x,y, legend='Data')
fig.line(x,y_predicted,color='red',legend= "Regression Line")
fig.yaxis.axis_label = 'Wind Direction During Rising Tide (deg)'
fig.xaxis.axis_label = 'Current Direction During Rising Tide (deg)'
show(fig)
