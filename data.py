import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("cityhall_clean.csv")
#print(df.head)
#print(df.describe())
x = df.iloc[:,1:].values
#print(x)
t = df.iloc[:,1].values.astype(float) 
#print(t)

















'''
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("cityhall_clean.csv")
#print(df.head)
#print(df.describe)
x = df.iloc[:,1:].values
#print(x)
t = df.iloc[:,1].values.astype(float) 
#print(t)

df.plot(kind='bar',x='hour',y='Total_Demand_KW')
plt.savefig('barchart01.png')
'''