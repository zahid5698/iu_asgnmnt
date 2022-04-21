import pandas as pd
import unittest
from Assignment_iubh import MathO #MathO() class has been imported from main file

df_t1 = pd.DataFrame({'A' : [1, 2], 'B' : [3, 4]})  #1st dummy table declared
df_t2 = pd.DataFrame({'C' : [2, 1], 'D' : [4, 3]})  #2nd dummy table declared
                                                    #result of joining two dummy table
df_result = pd.DataFrame({'A' : [1, 2], 'B' : [3, 4],'C' : [2, 1], 'D' : [4, 3]})

"""
class MathO():
    def __init__(self, df_1, df_2):
        self.df1 = df_1
        self.df2 = df_2

    def join_func(self):
        frames = [self.df1, self.df2]
        z=pd.concat(frames, axis=1, join='inner')
        return z

    def sqr_dev(self,p):
        x = MathO(df,df1)
        z = x.join_func()
        for c in cols:
            z[c]= round(((z[p]-z[c])**2), 2)
        z = z.iloc[:, :-2]
        return z
"""
# a custom class declared that contain unittest.Testcase as parameter
class UnitTestMathO(unittest.TestCase):
    def test_sqrdev(self):  # sqr_dev of MathO() class will be tested
        # an object is defined from MathO class with two dummy table as input parameter
        x = MathO(df_t1,df_t2) 
        result = x.join_func() #join_func() of MathO() is called
        # assertTrue method has been used to test the result which returns True/Flase
        #if the test fails, it will return a string
        self.assertTrue(result.equals(df_result), "The result should be:{}".format(df_result))

if __name__ == "__main__":
    unittest.main()
