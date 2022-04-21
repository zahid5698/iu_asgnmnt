import sqlite3
import sqlalchemy as db
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Float, Integer, Table, select, MetaData
from sqlalchemy.orm import sessionmaker
import csv
import pandas as pd
from math import sqrt
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas_bokeh
from bokeh.plotting import figure, output_file, show

# engine is created to interact with sqlite datbase
engine = create_engine('sqlite:///assignment.sqlite1', echo=True)
#session object created to bind with the engine
Session = sessionmaker(bind=engine)
session = Session()
metadata = MetaData()           #metadata contains table object

Ideal = declarative_base()      #object created from dectlerative_base class
class Listing(Ideal):       #           
    __tablename__ = 'idealfun'  # empty table created named 'idealfun' to contain 50 ideal functions
    id = Column(Integer, primary_key=True)  #primary key column declared
    x = Column(Integer)         # x column to contain integer data
    y1 = Column(Float)          #other columns to contain float data
    y2 = Column(Float)
    y3 = Column(Float)
    y4 = Column(Float)
    y5 = Column(Float)
    y6 = Column(Float)
    y7 = Column(Float)
    y8 = Column(Float)
    y9 = Column(Float)
    y10 = Column(Float)
    y11 = Column(Float)
    y12 = Column(Float)
    y13 = Column(Float)
    y14 = Column(Float)
    y15 = Column(Float)
    y16 = Column(Float)
    y17 = Column(Float)
    y18 = Column(Float)
    y19 = Column(Float)
    y20 = Column(Float)
    y21 = Column(Float)
    y22 = Column(Float)
    y23 = Column(Float)
    y24 = Column(Float)
    y25 = Column(Float)
    y26 = Column(Float)
    y27 = Column(Float)
    y28 = Column(Float)
    y29 = Column(Float)
    y30 = Column(Float)
    y31 = Column(Float)
    y32 = Column(Float)
    y33 = Column(Float)
    y34 = Column(Float)
    y35 = Column(Float)
    y36 = Column(Float)
    y37 = Column(Float)
    y38 = Column(Float)
    y39 = Column(Float)
    y40 = Column(Float)
    y41 = Column(Float)
    y42 = Column(Float)
    y43 = Column(Float)
    y44 = Column(Float)
    y45 = Column(Float)
    y46 = Column(Float)
    y47 = Column(Float)
    y48 = Column(Float)
    y49 = Column(Float)
    y50 = Column(Float)

Ideal.metadata.drop_all(bind=engine)    # delete all previous data before each run
Ideal.metadata.create_all(engine)       # create new table at each run        


try:        #try to catch error if the csv file not found
            #open the csv file containd ideal functions as csv_file
    with open('ideal_fun.csv', encoding='utf-8', newline='') as csv_file:
            #read the data in csv file as dictionary, as columns are the keys
        csvreader = csv.DictReader(csv_file, quotechar='"',)
            #loop through the columns to read data
        listings = [Listing(**row) for row in csvreader]
            #add all data to the table
        session.add_all(listings)
            #if csv file not found, raises error
except FileNotFoundError:
    print('File not found!!', 'ideal_fun.csv')

#same as before create empty table to contain training data
Trng = declarative_base()
class Training(Trng):
            
    __tablename__ = 'trngfun'   #table name is trngfun
    id = Column(Integer, primary_key=True)
    x = Column(Integer)
    y1 = Column(Float)
    y2 = Column(Float)
    y3 = Column(Float)
    y4 = Column(Float)
            #clear previous data from the table and create new table in each run
Trng.metadata.drop_all(bind=engine)
Trng.metadata.create_all(engine)

try:        #try to open csv file containing training data
            #open the csv file contained trianing data as csv_file
    with open('trainingdata.csv', encoding='utf-8', newline='') as csv_file:
        csvreader = csv.DictReader(csv_file, quotechar='"',)
            #read the data in csv file as dictionary, as columns are the keys
        listings = [Training(**row) for row in csvreader]
            #add all data to the table
        session.add_all(listings)
            #if csv file not found, raises error
except FileNotFoundError:
    print('File not found!!', 'trainingdata.csv')

#table created and data loaded from csv file for test data, same as before
Tst = declarative_base() 
class Test(Tst):
    __tablename__ = 'testfun'
    id = Column(Integer, primary_key=True)
    x = Column(Integer)
    y1 = Column(Float)

Tst.metadata.drop_all(bind=engine)
Tst.metadata.create_all(engine)

try:      #try to open the csv file containi
    with open('testdata.csv', encoding='utf-8', newline='') as csv_file:
        csvreader = csv.DictReader(csv_file, quotechar='"',)
        listings = [Test(**row) for row in csvreader]
            #add all data to the table
        session.add_all(listings)
except FileNotFoundError:
    print('File not found!!', 'testdata.csv')

#session saved and closed        
session.commit()    
session.close()

#all data of the threetables are rounded to 2 decimal using pandas
df = round(pd.read_sql_table('idealfun', con=engine), 2)
df1 = round(pd.read_sql_table('trngfun', con=engine), 2)
df2 = round(pd.read_sql_table('testfun', con=engine), 2)

#all the three tables are printed to screen using pandas dataframe
print("\n Ideal function's data: df(50 colx100 rows\n")
print(df)
print("\n Tainnng function's data: df1(4 colx100 rows\n")
print(df1)
print("\n Test function's data: df2(01 colx100 rows\n")
print(df2)

#columns of training table are renamed before joining to ideal functions
n = 2           #n=2, to avoid renaming id and x columns
while n < 6:    #column no 2,3,4 and 5 are renamed
    df1.rename(columns={df.columns[n]: 'ya{}'.format(n-1)}, inplace=True)
    n=n+1
print("\n Training function after columns renamning: df1\n")
print(df1)      #training data printed after column renaming
                #print the jointed table
print("\nAfter joining two data frames in panda:df3 = df+df1\n")
                #a list called 'cols' is created containing column names
cols = ['y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8','y9','y10',
    'y11', 'y12', 'y13', 'y14','y15','y16', 'y17', 'y18', 'y19', 'y20',
    'y21', 'y22', 'y23', 'y24', 'y25', 'y26', 'y27', 'y28', 'y29', 'y30',
    'y31', 'y32', 'y33', 'y34', 'y35', 'y36', 'y37', 'y38', 'y39', 'y40',
    'y41', 'y42', 'y43', 'y44', 'y45', 'y46', 'y47', 'y48', 'y49', 'y50']

#a class, MathO() is declared that takes two dataframe as input paramenter/
# and join them and clculate square deciations between columns 
class MathO():  
    def __init__(self, df_1, df_2): #initialize the object attributes
        self.df1 = df_1             # 1st dataframe
        self.df2 = df_2             # 2nd dataframe

    def join_func(self):            # method is declared to join two data frames
        frames = [self.df1, self.df2]   # frames object contains a list of two dataframes
        # concat function() join two dataframes, cloumnwise and type of joint is inner
        z=pd.concat(frames, axis=1, join='inner')
        return z                    # returns one dataframe after joining 

    def sqr_dev(self,p):    # method is created to calculate squared deviations between two columns                
        x = MathO(df,df1)   # an object of MathO() is created
        z = x.join_func()   # returns a jointed table
        #calculate squared deviations between a fixed and a variable column 
        #loop through each column to get the deviations between every y pair values
        #put the result in the same location of the variable column 
        
        for c in cols:
            z[c]= round(((z[p]-z[c])**2), 2) 
        z = z.iloc[:, :-6]      # discard last 6 column from the table
        return z
   
x = MathO(df,df1)   # object x is created of MathO(), with ideal and training
                    # data table as input parameters
df3 = x.join_func() # ideal and training data table are jointed
#print(df3)

def sum_columns(p):         # functions is defined to sum all the alues of a column
    y=p.copy(deep=True)     # copy the whole table to perform the sum operations
    for c in cols:
        y[c] = y[c].sum()   # calculate the sum of the values of a column using sum()
    y=y.drop(y.index[1:100], axis=0) # drop the index row
    y=y.iloc[:, 2:]         # keep the first 2 columns of the table
    return y

print("\n1. deviations squred between ideal(50) func & 1st training func(ya1):df4\n")
# bring out the sqr deviation between all 50 ideal function and 1st training data
df4 = x.sqr_dev('ya1')
print(df4)
# bring out the sqr deviation between all 50 ideal function and 2nd training data
print("\nY 2. deviations squred between ideal(50) func & 2nd training func(ya2):df5\n")
df5 = x.sqr_dev('ya2')
print(df5)
# bring out the sqr deviation between all 50 ideal function and 3rd training data
print("\nY 3. deviations squred between ideal(50) func & 3rd training func(ya3):df6\n")
df6 = x.sqr_dev('ya3')
print(df6)
# bring out the sqr deviation between all 50 ideal function and 4th training data
print("\nY 4. deviations squred between ideal(50) func & 4th training func(ya4):df7\n")
df7 = x.sqr_dev('ya4')
print(df7)

df8 = sum_columns(df4)  # sum the columns of df4
df9 = sum_columns(df5)  # sum the columns of df5
df10 = sum_columns(df6) # sum the columns of df6
df11 = sum_columns(df7) # sum the columns of df7

print("\n Afeter joining four table(rows) of deviation sums:df12\n ")
frames = [df8, df9, df10, df11]     # list of 4 tables
df12 = pd.concat(frames, axis=0)    # join 4 tables
print(df12)


print("\n Column values corresponding to min values in row:df15\n")
cols = [df12.idxmin(axis=1)] # sort out the column index containing the min valu in each column
for c in cols:  
    df15 = (df[c])

print(df15)         # print the original column values related to the column index as got above

print("\n Selected functions with x column added::df16\n")

df16 = df15.copy(deep=True) # copy the df15 table to perform further operations 
data = list(range(1, 101, 1))
df16.insert(0, 'x', data)   # insert x column with values 1-100 with df16 table
print(df16)

print("\nSquare of y deviations correspondind to Ideal functions that minimizes the y\
    deviations for four trg func:df22")
df18 = df4[df15.columns[0]] # get the squared deviations corresponding to the 1st column index of table df15
df19 = df5[df15.columns[1]] # get the squared deviations corresponding to the 2nd column index of table df15
df20 = df6[df15.columns[2]] # get the squared deviations corresponding to the 3rd column index of table df15
df21 = df7[df15.columns[3]] # get the squared deviations corresponding to the 4th column index of table df15

frames = [df18, df19, df20, df21]
df22 = pd.concat(frames, axis=1) # join the above four columns to form a table
print(df22)     

print("\n Max deviation in each column selected for four trg func:df23\n")

df23 = df22.max() # bring out the maximun deviation values in each column of df22
print(df23)

print("\n Predicted values: df24\n")

# lin_reg() function is defined to find out the curve of best fit
# and find out the perdicted values based on four selected ideal functions 
def lin_reg(p):
    # take X axis value from table of four selected ideal fucntions df16
    X = np.array([df16.iloc[:,0]]).reshape((-1, 1)) 
    # take y axis values from the column p of table df16
    y = np.array(df16.iloc[:,p]) 
    model = LinearRegression() # model object drfined from LinearRegression class
    model.fit(X,y)             # relevant co-efficients are calculated for curve of best fit
    y_pred = model.predict(X)  # values for predicted curve is found out from curve of best fit
    return y_pred

# array will be formed with the predicted values based on selected ideal function 
def convert_arr(x,y):   
    z = lin_reg(x)      # lin_reg function is called with input column parameter,x
    z = pd.DataFrame(z,columns = [y])
    return z            # return a data frame with predicted values

df24 = convert_arr(1,df22.columns[0]) # array is formed with predicted values based on 1st selected function
df25 = convert_arr(2,df22.columns[1]) # array is formed with predicted values based on 2nd selected function
df26 = convert_arr(3,df22.columns[2]) # array is formed with predicted values based on 3rd selected function
df27 = convert_arr(4,df22.columns[3]) # array is formed with predicted values based on 4th selected function

frames = [df24, df25, df26, df27]
df28 = pd.concat(frames, axis=1)    #above four tables are jointed together


df29 = df2.copy(deep = True)        # table for test data is copied and columns are renamed
df29.rename(columns={df.columns[2]: 'y_test'},inplace=True)
df30 = df29.drop(columns=["x","id"]) # x and id column is dropped
frames = [df28, df30]                # tables for predicted values, actual values and test data are jointed
df31 = round((pd.concat(frames, axis=1,join = 'inner')),2)
print("\n Final table with predicted values of corresponding selected functions:df31\n")
print(df31)                         # print table with predicted values

df32 = df31.copy(deep = True)

# deviation between test data and predicted data are calculated
print("\n Deviations from actual and predicted values:df32\n")
n=0
while n < 4:
    # takes the absolute value of difference between test data and all the four sets of predicted values
    df32.iloc[:,n]= abs(df32.iloc[:,4]-df32.iloc[:,n]) 
    n = n+1
print(df32)
# Calculate the maximum deviations between test data an predicted data
print("\n Maximum deviations between actual and predicted values of corresponding functions:df34\n")
df33 = df32.copy(deep=True)
df34 = df33.max()   # takes the maximum deviation valu from table df32
print(df34)

print("\n")
n = 0
while n<4:
    # Compares the max deviation between test data & prediacted data and
    # maximum deviation between selected functions and training data
    if df34[n] <= df23[n]*sqrt(2):
        print(df34.index[n], ":is a suitable function:\n")
    else:
        print(df34.index[n], ":is not a suitable function:\n")
    n = n+1

print ("\n Plt show : df35\n")
df35 = df31.copy(deep=True)     # make a copy of table of predicted data
data = list(range(1, 101, 1))   # inser x column values to be 1-100
df35.insert(0, 'x', data)
                                #renames  columns name of table df35
df35.columns = ['x', '1st Func', '2nd_Func', '3rd_Func', '4th_Func', 'Test Func']
print(df35)  

# plot() function is defined to plot multuple lines for 
# test data and predicted values based on pandas bokeh
def plot(p, q):
    output_file("gfg{}.html".format(p))             #output file is stored in a .html file
    graph = figure(title = "Bokeh Multi Line Graph")# figure-object is defined with title
    xs = [df35.iloc[:, 0], df35.iloc[:, 0]]         # two x axis data is given
    ys = [df35.iloc[:, p], df35.iloc[:, 5]]         # two y axis data is given
    graph.xaxis.axis_label = "x-axis"               # x axis and y axis is labeled
    graph.yaxis.axis_label = "Test Data & {} Predicted Data".format(q)
    line_color = 'red', 'blue'                      # two different color is given
    graph.multi_line(xs, ys, line_color=line_color) # final graph is plotted with multiple lines
    return graph
  
show(plot(1, "1st")) # 1st graph is plotted for test data and 1st selected function
show(plot(2, "2nd")) # 2nd graph is plotted for test data and 2nd selected function
show(plot(3, "3rd")) # 3rd graph is plotted for test data and 3rd selected function
show(plot(4, "4th")) # 4th grph is plotted for test data and 4th selected function
