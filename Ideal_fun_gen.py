import math
import numpy as np
import pandas as pd
import csv
print("FUNCTION : START Creating a CSV file")

x = np.arange(1, 101, 1)        # set x values to 1-100
file = open('data.csv','w', newline = "") #open data.csv file in write mode
csvwriter = csv.writer(file)    #create csvwriter object to write on csv file
csvwriter.writerow(x)           #write 2 rows of values as 1-100 in csv file
csvwriter.writerow(x)

#data for  1st 10xideal function generated in the below code
n = 1
c = 1
y1 = []     #empty list declared
while (n<11 and c<11): # loop 10 times with c and n values 1-10
    y1 = c*np.sin(n*x) # 1st ideal function taken to be y = c.sinx
    y1 = [round(i, 2) for i in y1] # all the y values are rounded upto 2 decimals
    csvwriter.writerow(y1)  #write the values on csv file
    n = n+1
    c = c+1
file.close()    #csv file sved and closed

#data for  2nd 10xideal function generated in the below code
file = open('data.csv','a', newline = "") #data.csv file opened in append mode
csvwriter = csv.writer(file)    #csvwrite object is created
n = 1
y2 = []         #empty list declared
while (n<11):   # loop 10 times with n values 1-10
    y2 = x**2/n #2nd ideal function taken to be y = x**2/n
    y2 = [round(i, 2) for i in y2] # y values are rounded
    csvwriter.writerow(y2) # y values written to csv file
    n= n+1

#data for 3rd 10xideal function generated in the below code
n = 1
c = 1
y3 = []
while (n<11 and c <22): # loop 10 times with n = 1-10 and c = 1-21
    y3 = x**2 + n*x +c  # 3rd ideal function take nto be y = x**2+n*x+c
    y3 = [round(i, 2) for i in y3] # y vlaues rounded
    csvwriter.writerow(y3)      #y values written on csv file
    n= n+1
    c = c+2

#data for 4th 10xideal function generated in the below code
n = 1
c = 1
y4 = []
while (n<11 and c<11):  # loop 10 times c & n values 1-10
    y4 = n*x + c        # 4th ideal function taken to be y=n*x+c
    y4 = [round(i, 2) for i in y4]
    csvwriter.writerow(y4)  # y values written on csv file
    n = n+1
    c = c+1

#data for 5th 10xideal function generated in the below code   
n = 1
c = 1
y5 = []         #empty list created
while (n<11 and c<110): # loop 10 times
    y5 = n*np.log10(x) + c  # 5th ideal function taken to be n*log10(x)+c
    y5 = [round(i, 2) for i in y5]
    csvwriter.writerow(y5)  #y values written to csv file
    n = n+1
    c = c+10
file.close()    #csv file saved and closed

df = pd.read_csv('data.csv') #data.csv file converted to pandas dataframe
df1 = df.T      # row and column are transposed in pandas dataframe
#column names are addded to have x values and 50 x y-values for fifty ideal functions
df1.columns = ['x','y1', 'y2', 'y3','y4','y5','y6','y7', 'y8','y9','y10',
    'y11', 'y12', 'y13','y14','y15','y16', 'y17','y18','y19','y20',
    'y21', 'y22', 'y23','y24','y25','y26','y27','y28','y29','y30',
    'y31','y32','y33','y34','y35','y36','y37','y38','y39','y40',
    'y41','y42','y43','y44','y45','y46','y47','y48','y49','y50']

print(df1)  # data for 50 ideal functions is printed in table fotmat

# above table is re-written in another 'ideal_fun.csv' file
df1.to_csv('ideal_fun.csv', index = False)  
print("--------END---------")