import pandas as pd 
#read file
data = pd.read_csv("cityhall.csv")
#print(data.head)
#select column 1
dt = data.iloc[:,0]
#print(dt)
#split column 
dt = dt.str.split(' ', expand=True)
#print(dt)

#change data type
dt.iloc[:,0] = dt.iloc[:,0].astype('datetime64[ns]')
dt.iloc[:,1] = dt.iloc[:,1].astype('datetime64[ns]')

#print(dt.iloc[:,0].dt.year)
df = pd.concat([dt.iloc[:,0].dt.year,
                dt.iloc[:,0].dt.month,
                dt.iloc[:,0].dt.dayofweek,
                dt.iloc[:,1].dt.hour]
                , axis = 1)

data = pd.concat([data, df], axis = 1)
print(data.head)


#drop out
data = data.drop(['DateTime_Measured'], axis=1)
data.columns = ['Total_Demand_KW', 'year', 'month', 'dayofweek', 'hour']

#save
data.to_csv('cityhall_clean.csv', index=False)
