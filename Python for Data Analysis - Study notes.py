#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame


# In[ ]:


#Generate some random data
data = np.random.randn(3)
print(f"""data{data}, 
data x 10{data * 10}, 
data+data{data + data}""")


# In[ ]:


data1 = [1, 5, 8.5, 2, 0, 1]
arr1 = np.array(data1)
arr1


# In[ ]:


data2= [[1, 5, 8.5], [2, 0, 1]]
arr2= np.array(data2)
arr2


# In[ ]:


arr = np.array([1, 2, 3, 4, 5, 6, 7])
arr.dtype


# In[ ]:


float_arr = arr.astype(np.float64)
float_arr.dtype


# In[ ]:


numeric_strings = np.array(['1.25', '-9.6', '42'], dtype=np.string_)
numeric_strings.astype(float)


# In[ ]:


#Arithmetic with NumPy Arrays
arr = np.array([[1., 2., 3.], [4., 5., 6.]])
arr2 = np.array([[0., 4., 1.], [7., 2., 12.]])

arr2>arr


# In[ ]:


#Basic Indexing and Slicing
arr = np.arange(10)
arr[5:8] = 12
arr


# In[ ]:


arr_slice = arr[5:8]
arr_slice[1] = 12345
print(arr_slice, arr)


# In[ ]:


arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d[0, 2]


# In[ ]:


arr3d = np.array([[[1, 2, 3,], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d


# In[ ]:


old_values = arr3d[0].copy()
arr3d[0] = 42
arr3d


# In[ ]:


arr3d[0] = old_values
arr3d


# In[ ]:


arr3d[1, 0]


# In[ ]:


arr2d[1, :2]


# In[ ]:


arr2d[:, :1]


# In[ ]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
names


# In[ ]:


data


# In[ ]:


names == 'Bob'


# In[ ]:


data[names == 'Bob', 3]


# In[ ]:


names != 'Bob'


# In[ ]:


data[~(names == 'Bob')]


# In[ ]:


mask = (names == 'Bob') | (names == 'Will')
mask


# In[ ]:


data[mask]


# In[ ]:


data[data < 0] = 0
data


# In[ ]:


data[names != 'Joe'] = 7
data


# In[ ]:


arr = np.empty((8, 4))
for i in range(8):
    arr[i] = i
arr


# In[ ]:


arr[[-3, -5, -7]]


# In[ ]:


arr = np.arange(32).reshape((8, 4))
arr


# In[ ]:


arr.swapaxes(0, 1)


# In[ ]:


#Universal Funtions: Fast Element-Wise Array Functions 
# ufunc is a function that performs element wise operations on data in ndarrays like vectorized wrappers for simple functions


# In[ ]:


arr = np.arange(10)
arr


# In[ ]:


#sqrt and exp are unary ufuncs
np.sqrt(arr)


# In[ ]:


np.exp(arr)


# In[ ]:


#ufuncs that take two arrays and return a single result are binary ufuncs
x = np.random.randn(8)
y = np.random.randn(8)
x


# In[ ]:


y


# In[ ]:


np.maximum(x, y)


# In[ ]:


#a ufunc can return multiple arrays arrays.modf is one example it returns the fractional and integral parts of a floating-point array
arr = np.random.randn(7) * 5
arr


# In[ ]:


remainder, whole_part = np.modf(arr)
remainder


# In[ ]:


whole_part


# In[ ]:


#ufuncs accep an optional out argument that allows them to operate in-place on arrays
arr1 = np.random.randn(7) * 5
arr1


# In[ ]:


arr


# In[ ]:


np.sqrt(arr)


# In[ ]:


arr


# In[ ]:


np.sqrt(arr1)


# In[ ]:


arr1


# In[ ]:


np.sqrt(arr, arr1)


# In[ ]:


np.sqrt(arr1, arr)


# In[ ]:


np.sqrt(arr, arr1)


# In[ ]:


np.sqrt(arr1, arr)


# In[ ]:


arr


# In[ ]:


arr1


# In[ ]:


np.sqrt(arr, None)


# In[ ]:


arr


# In[ ]:


#Array-Oriented programming with arrays
points = np.arange(-5, 5, 0.01)


# In[ ]:


xs, ys = np.meshgrid(points, points)
ys


# In[ ]:


xs


# In[ ]:


z = np.sqrt(xs ** 2 + ys ** 2)
z


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.imshow(z, cmap=plt.cm.gray); plt.colorbar(); plt.title("Image plot of $sqrt{x^2 + y^2}$ for a grid of values")


# In[ ]:


#numpy.where is a vectorized version of the ternary expression x if condition else y. If we had a boolean array and two arrays of values.
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])
result = [(x if c else y)
         for x, y, c in zip(xarr, yarr, cond)]
result
#This has multiple problems. It will not be very fast for large arrays. It will not work with multidimensional arrays.


# In[ ]:


#with np.where you can write this very concisely 
result = np.where(cond, xarr, yarr)
result


# In[ ]:


arr = np.random.randn(4, 4)
arr


# In[ ]:


arr > 0


# In[ ]:


np.where(arr > 0, 2, -2)


# In[ ]:


np.where(arr > 0, 2, arr) # set only positive values to 2


# In[ ]:


#Mathmatical and Statistical Methods
# you can use aggregations(often called reductions) like sum, mean, std(standard deviation) either by calling the array or
# using the top-level NumPy function
# here I generate some normally distributed reandom data and compute some aggregate statistics
arr = np.random.randn(5, 4)
arr


# In[ ]:


arr.mean()


# In[ ]:


np.mean(arr)


# In[ ]:


arr.sum()


# In[ ]:


arr.mean(axis=1)


# In[ ]:


arr.sum(axis=0)


# In[ ]:


#other methods like cumsum and cumprod do not aggregate instead produce an array of the intermediate results
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7])
arr.cumsum()


# In[ ]:


#in multidimensional arrays accumulation functions like cumsum return an array of the same size but with the partial 
#aggregates computed along the indicated axis according to each lower dimensional state
arr = np.array([[1, 2, 3], [3, 4, 5], [6, 7, 8]])
arr


# In[ ]:


arr.cumsum(axis=0)


# In[ ]:


arr.cumsum(axis=1)


# In[ ]:


arr.cumprod(axis=1)


# In[ ]:


#Methods for Boolean Arrays
#Boolean values are coerced to 1(T) and 0(F) in the preceding methods. sum is often used to count True values in a bool array.
arr = np.random.randn(100)
(arr > 0).sum()


# In[ ]:


#Any and all are useful in boolean arrays. any tests whether one or more values are T all checks if every value is T
bools = np.array([False, False, True, False])
bools.any()


# In[ ]:


bools.all()
#with non boolean arrays non-zero elements evaluate to True


# In[ ]:


#Sorting
#NumPy arrays can be sorted in place with the sort method just like Pythons built in list type
arr = np.random.randn(6)
arr


# In[ ]:


arr.sort()
arr


# In[ ]:


#you can sort each one-dimensional section of values in a multidimentional array in-place along an axis by passing the axsi number to sort
arr = np.random.randn(5, 3)
arr


# In[ ]:


arr.sort(1)


# In[ ]:


arr


# In[ ]:


arr.sort(0)
arr


# In[ ]:


large_arr = np.random.randn(1000)
large_arr.sort()
large_arr[int(0.05 * len(large_arr))] # %5 quantile


# In[ ]:


#Unique and Other Set Logic
#NumPy has some basic set operations for one dimensional ndarrays. A commondly used one is np.unique which return the 
#sorted unique values in an array
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.unique(names)


# In[ ]:


ints = np.array([3, 3, 3, 3, 2, 2, 1, 1, 4, 4])
np.unique(ints)


# In[ ]:


#contrast np.unique with the pure Python alternative
sorted(set(names))


# In[ ]:


#np.in1d tests membership of the values in one array in another, returning a boolean array:
values = np.array([6, 0, 0, 3, 2, 5, 6])
np.in1d(values, [2, 3, 6])


# In[ ]:


#File Input and Output with Arrays
#NumPy is able to save and load data to and from disk either in text or binary format
#np.save and np.load arrays are saved by default in an uncompressed raw binary format with file extension .npy
arr = np.arange(10)
np.save('some_array', arr)


# In[ ]:


#If the file path does not already end in .npy the extension will be appended the array on disk can then be loaded with np.load
np.load('some_array.npy')


# In[ ]:


#You save multiple arrays in an uncompressed archive using np.savez and passing arrays as keyword arguments
np.savez('array_archive.npz', a=arr, b=arr)


# In[ ]:


arch = np.load('array_archive.npz')
arch['a']


# In[ ]:


#if your data compresses well you may wish to use numpy.savez_compressed instead:
np.savez_compressed('array_compressed.npz', a=arr, b=arr)


# In[ ]:


#Linear Algebra 
x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
x


# In[ ]:


y


# In[ ]:


x.dot(y)
#equivalent with np.dot(x, y)


# In[ ]:


y.dot(x)


# In[ ]:


#A matrix product between a two dimensional array and a suitably sized one dimensional array results in a one dimensional array
np.dot(x, np.ones(3))


# In[ ]:


#The @ symbol works as an infix operator(in the middle :p) that performs matrix multiplication
x @ np.ones(3)


# In[ ]:


#numpy.linalg has a standard set of matrix decompositions and things like inverse and determinant. These are implemented
#under the hood via the same industry standard linear algebra libraries used in other languages like MATLAB and R
from numpy.linalg import inv, qr
X = np.random.randn(5, 5)
mat = X.T.dot(X)
#The expression X.T.dot(X) computes the dot product of X with its transpose X.T (flip on the diagonal)
inv(mat)


# In[ ]:


#pseudorandom number generation
#numpy.random module supplements the built in python random with fuctions for efficiently generation whole arrays
# you can get a 4x4 array of samples from the standard normal distribution using normal:
samples = np.random.normal(size=(4, 4))
samples


# In[ ]:


from random import normalvariate
N = 1000000
get_ipython().run_line_magic('timeit', 'samples = [normalvariate(0, 1) for _ in range(N)]')


# In[ ]:


get_ipython().run_line_magic('timeit', 'np.random.normal(size=N)')


# In[ ]:


#you can change numppy's random number generator seed using np.random.seed
np.random.seed(1234)
#The data generation functions in numpy.random use a global random seed. to avoid global use numpy.random.RandomState
rng = np.random.RandomState(1234)
rng.randn(10)


# In[ ]:


#example: random walks
#pure python way of doing a simple +1 -1 walk with 1000 steps
import random 
position = 0
walk = [position]
steps = 1000
for i in range(steps):
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)


# In[ ]:


plt.plot(walk[:100])


# In[ ]:


#walk is just the cumulative sum of the random steps and could be evaluated as an array expression.
#use np.random module to draw 1000 coinflips at once set these to 1 and -1 and compute the cumulative sum
nsteps = 1000
draws = np.random.randint(0, 2, size=nsteps)
steps = np.where(draws > 0, 1, -1)
walk = steps.cumsum()
walk.min()


# In[ ]:


walk.max()


# In[ ]:


#argmax return the first index of the maximum value in the boolean array (T is the max value)
(np.abs(walk) >= 10).argmax()
#argmax is not always efficient because it will do a full scan of the array


# In[ ]:


#simulating many random walks at once
#we can compute the cumulative sum across the rows to compute all 5,000 random walks in one shot:
nwalks = 5000
nsteps = 1000
draws = np.random.randint(0, 2, size=(nwalks, nsteps)) # 0 or 1
steps = np.where(draws > 0, 1, -1)
walks = steps.cumsum(1)
walks


# In[ ]:


#now we can compute the maximum and minimum values over all of the walks
walks.max()


# In[ ]:


walks.min()


# In[ ]:


#out of these walks lets compute the minimum crossing time to 30 or -30 not all reach 30 so we use any method
hits30 = (np.abs(walks) >= 30).any(1)
hits30


# In[ ]:


hits30.sum()


# In[ ]:


crossing_times = (np.abs(walks[hits30]) >= 30).argmax(1)
crossing_times


# In[ ]:


crossing_times.mean()


# In[ ]:


steps = np.random.normal(loc=0, scale=0.25, size=(nwalks, nsteps))


# In[ ]:


steps.mean()


# In[ ]:


#chapter 5 page 123
import pandas as pd
from pandas import Series, DataFrame
#introduction to pandas data structures - series and dataframe
#Series is a one dimensional array containing a sequence of values + associated array of data labels called its index
obj = Series([4, 7, -5, 3])
obj


# In[ ]:


#you can get the array representation and index object of the series via its values and index attributes respectively:
obj.values


# In[ ]:


obj.index #like range(4)


# In[ ]:


#if its desirable to creat a series with an index identifying each data point with a label you can do this:
obj2 = Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
obj2


# In[ ]:


obj2.index


# In[ ]:


obj2['a']


# In[ ]:


obj2['d'] = 6
obj2[['c', 'a', 'd']]


# In[ ]:


obj2[obj2 > 0]


# In[ ]:


obj2*2


# In[ ]:


np.exp(obj2)


# In[ ]:


#Another way to think about a series is as a fixed-length, ordered dict, as it is a mapping of index values to data values
'b' in obj2


# In[ ]:


'e' in obj2


# In[ ]:


#should you have data contained in a python dict, you can create a series from it by passing the dict
sdata = {'Ohio' : 35000, 'Texas' : 71000, 'Oregon' : 16000, 'Utah' : 5000}
obj3 = Series(sdata)
obj3


# In[ ]:


states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = Series(sdata, index = states)
obj4


# In[ ]:


pd.notnull(obj4)


# In[ ]:


obj4.isnull()


# In[ ]:


obj3


# In[ ]:


obj4


# In[ ]:


obj3+obj4


# In[ ]:


obj4.name = 'population'
obj4.index.name = 'state'
obj4


# In[ ]:


obj


# In[ ]:


obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']
obj


# In[ ]:


#dataframe - has both a row and column index; it can be thought of as a dict of Series all sharing the same index
#There are many ways to construct a DataFrame, one of the most common is from a dict of equal-lenght lists or NumPy arrays
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
       'year' : [2000, 2001, 2002, 2001, 2002, 2003],
       'pop' : [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
#resulting dataframe will have its index assigned automatically as with Series and the columns are placed in sorted order:
frame = DataFrame(data)
frame


# In[ ]:


frame.head()


# In[ ]:


#if you specify a sequence of columns the DataFrame's columns will be arranged in that order:
DataFrame(data, columns=['year', 'state', 'pop'])


# In[ ]:


frame2 = DataFrame(data, columns=['year', 'state', 'pop', 'debt'],
                  index=['one', 'two', 'three', 'four', 'five', 'six'])
frame2


# In[ ]:


#a column can be retrieved as a series either by dict-like notation or by attribute
frame2['state']


# In[ ]:


frame2.year


# In[ ]:


frame2.loc['three']


# In[ ]:


frame2['debt'] = 16.5
frame2


# In[ ]:


frame2['debt'] = np.arange(6.)
frame2


# In[ ]:


#when you are assigning lists or arrays to a column, the values length must match the length of the DataFrame.
#If you assign a Series, its labels will be realigned exactly to the DataFrame's index inserting missing values in any holes
val = pd.Series([-1.2, -1.5, -1.7], index = ['two', 'four', 'five'])

frame2['debt'] = val
frame2


# In[ ]:


# Assigning a column that doesn't exist will create a new column. The del keyword will delete a column
frame2['eastern'] = frame2.state == 'Ohio'
frame2


# In[ ]:


del frame2['eastern']

frame2.columns


# In[ ]:


population = {'Nevada': {2001: 2.4, 2002: 2.9},
      'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
frame3 = DataFrame(population)
frame3


# In[ ]:


#you can transpose the DataFrame (swap rows and columns) with similar syntax to a NumPy array:
frame3.T


# In[ ]:


#If a DataFrame's index and columns have their name attributes set, these will also be displayed:
frame3.index.name = 'year'; frame3.columns.name = 'state'
frame3


# In[ ]:


frame3.values


# In[ ]:


#if the DataFrame's columns are different dtypes, the dtype of the values array will be chosen to accommodate all of the columns
frame2.values


# In[ ]:


#index objects 
#pandas index objects are responsible for holding the axis labels and other metadata. Any array you use with be converted
obj = Series(range(3), index=['a', 'b', 'c'])
index = obj.index
index


# In[ ]:


index[1:]


# In[ ]:


#index objects are immutable and thus can't be modified by the user:
index[1] = 'd'


# In[ ]:


#immutability makes it safer to share Index objects among data structures:
labels = pd.Index(np.arange(3))
labels


# In[ ]:


obj2 = Series([1.5, -2.5, 0], index=labels)
obj2


# In[ ]:


obj2.index is labels


# In[ ]:


frame3


# In[ ]:


#in addition to being array-like an index also behaves like a fixed size set
frame3


# In[ ]:


frame3.columns


# In[ ]:


'Ohio' in frame3.columns


# In[ ]:


2003 in frame3.index


# In[ ]:


2000 in frame3.index


# In[ ]:


#unlike python sets a pandas index can contain duplicate labels 
dup_labels = pd.Index(['foo', 'foo', 'bar', 'bar'])
dup_labels


# In[ ]:


#selections with duplicate labels will select all occurrences of that label
#essential functionality
#Reindexing
obj = Series(range(4), index=['d', 'b', 'a', 'c'])
obj.sort_index()


# In[ ]:


df = DataFrame([[1.4, np.nan], [7.1, -4.5],
               [np.nan, np.nan], [0.75, -1.3]],
              index = ['a', 'b', 'c', 'd'],
              columns = ['one', 'two'])
df


# In[ ]:


#Calling DataFrame's sum method returns a Series containing column sums:
df.sum()


# In[ ]:


#passing axis='columns' or axis=1 sums across the columns instead:
df.sum(axis='columns')


# In[ ]:


#NA values are excluded unless the entire slice (row or column in this case) is NA This can be disabled with the skipna option
df.mean(axis='columns', skipna=False)


# In[ ]:


df.mean(axis='columns')


# In[ ]:


#Some methods like idxmin and idxmax, return indirect statistics like the index value where the minimum or maximum values are attained
df.idxmax()


# In[ ]:


df.cumsum()


# In[ ]:


#another type of method is neither a reduction nor an accumulation. descrive is one such example, producing multiple summary stat
df.describe()


# In[ ]:


#on non-numeric data, describe produces alternative summary statistics 
obj = Series(['a', 'a', 'b', 'c'] * 4)
obj.describe()


# In[ ]:


#Indexing into a DataFrame is for retrieving one or more columns either with a single value or sequence
data = DataFrame(np.arange(16).reshape((4, 4)),
                index=['Ohio', 'Colorado', 'Utah', 'New York'],
                columns=['one', 'two', 'three', 'four'])
data


# In[ ]:


data['two']


# In[ ]:


data[['three', 'one']]


# In[ ]:


data[:2]


# In[ ]:


data[data['three']>5]


# In[ ]:


data<5


# In[ ]:


data[data<5]=0
data


# In[ ]:


data.loc['Colorado', ['two', 'three']]


# In[ ]:


data.iloc[2, [3, 0, 1]]


# In[ ]:


data.iloc[[1, 2], [3, 0, 1]]


# In[ ]:


#both indexing functions work with slices in addition to single labels or lists of labels 
data.loc[:'Utah', 'two']


# In[ ]:


data.iloc[:, :3][data.three > 5]


# In[ ]:


#Integer Indexes 
#pandas objects indexed by integers are wierd because of some differences with indexing semantics on built in python data 
#structures like lists and tupbles. you might not expect the following code to generate an error
ser = Series(np.arange(3.))
ser


# In[ ]:


ser[-1]


# In[ ]:


#In this case pandas could "fall back" on integer indexing but it's difficult to do this in general without introducing bugs
# Here we have an index containing 0, 1, 2, but inferring what the user wants (label based indexing or position based) is hard
ser


# In[ ]:


#with a non integer index there's no potential for ambiguity
ser2 = Series(np.arange(3.), index=['a', 'b', 'c'])
ser2[-1]


# In[ ]:


#to keep things consistent if you have an axis index containing integers, data selection will always be label-oriented.
#for more precise handling, use loc(for labels) or iloc(for integers)
ser.iloc[:1]


# In[ ]:


#Arithmetic and Data Alignment
s1 = Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
s2 = Series([-2.1, 3.6, -1.5, 4, 3.1], index=['a', 'c', 'e', 'f', 'g'])
s1


# In[ ]:


s2


# In[ ]:


s1 + s2


# In[ ]:


#In case of DataFrame, alignment is performed on both the rows and the columns
df1 = DataFrame(np.arange(9.).reshape((3, 3)), columns=list('bcd'), index=['Ohio', 'Texas', 'Colorado'])
df2 = DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])
df1


# In[ ]:


df2


# In[ ]:


df1 + df2


# In[ ]:


#If you add DataFrame objects with no column or row labels in common, the result will contain all nulls. 
df1 = DataFrame({'A': [1, 2]})
df2 = DataFrame({'B': [3, 4]})
df1


# In[ ]:


df2


# In[ ]:


df1 + df2


# In[6]:


#Arithmetic methods with fill values
#In arithmatic operations between differently indexed objects you might want to fill with a special value like 0 when an
#axis label is found in one object but not the other
df1 = DataFrame(np.arange(12.).reshape((3, 4)), columns = list('abcd'))
df2 = DataFrame(np.arange(20.).reshape((4, 5)), columns = list('abcde'))
df2.loc[1, 'b'] = np.nan
df1


# In[7]:


df2


# In[8]:


#adding these together makes nan for areas that don't overlap 
df1 + df2


# In[9]:


#using an add methon on df1 I pass df2 and an argument to fill_value:
df1.add(df2, fill_value=0)


# In[12]:


#these two statements are equivalent:
1/df1, df1.rdiv(1)


# In[15]:


#when indexing a Series or DataFrame you can also specify a different fill value
df1.reindex(columns=df2.columns, fill_value=0)


# In[17]:


#operations between DataFrame and Series. As with NumPy arrays and different dimensions, arithmetic between DataFrame and
#Series is also defined.First as a motivating example, consider the difference between a two-dimensional and one of its rows
arr = np.arange(12.).reshape((3, 4))
arr


# In[18]:


arr[0]


# In[19]:


arr - arr[0]


# In[20]:


#when we subtract arr[0] from arr, the subtraction is performed once for each row. This is referred to as broadcasting
frame = DataFrame(np.arange(12.).reshape((4, 3)),
                 columns = list('bde'),
                 index = ['Utah', 'Ohio', 'Texas', 'Oregon'])
series = frame.iloc[0]
frame


# In[21]:


series


# In[22]:


#by default arithmatic between DataFrame and Series matches the index of the Series on the DataFrame's columns broadcasting
frame - series


# In[23]:


#if an index value is not found in either the DataFrame's columns or the Series's index, the objects will be reindexes to
#form the union
series2 = Series(range(3), index=['b', 'e', 'f'])
frame + series2


# In[24]:


#if you want to instead broadcast over the columns matching on the rows you have to use one of the arithmetic methods
series3 = frame['d']
frame


# In[25]:


series3


# In[26]:


frame.sub(series3, axis='index')


# In[3]:


#function application and mapping
#NumPy ufuncs (element-wise array methods) also work with pandas objects:
frame = DataFrame(np.random.randn(4, 3), columns = list('bde'),
                 index = ['Utah', 'Ohio', 'Texas', 'Oregon'])
frame


# In[4]:


np.abs(frame)


# In[5]:


#Another frequent operation is applying a function on one-dimensional arrays to each column or row. DataFrame's apply method
# does exactly this:
f = lambda x: x.max() - x.min()
frame.apply(f)
#Here the function f, which computes the difference between the maximum and minimum of a Series, is invoked once on each
# Column in frame. The result is a Series having the columns of frame as its index.


# In[6]:


#If you pass axis='columns' to apply, the function will be invoked once per row instead:
frame.apply(f, axis='columns')
#Many of the most common array statistics (like sum and mean) are DataFrame methods, so using apply is not necessary.


# In[8]:


#The function passed to apply need not return a scalar value; it can also return a Series with multiple values:
def f(x):
    return Series([x.min(), x.max()], index=['min', 'max'])
frame.apply(f)


# In[11]:


#Element-wise Python functions can be used, too. Suppose you wanted to compute a formatted string from each floating-point
#value in frame. You can do this with apply map:
format = lambda x: '%.2f' % x
frame.applymap(format)


# In[12]:


#The reason for the name applymap is that Series has a map method for applying an element-wise function:
frame['e'].map(format)


# In[13]:


#Sorting and Ranking
#Sorting a dataset by some criterion is another important built-in operation. To sort lexicographically by row or column index
#use the sort_index method, which returns a new, sorted object:
obj = Series(range(4), index=['d', 'a', 'b', 'c'])
obj.sort_index()


# In[2]:


#With a DataFrame, you can sort by index or either axis:
frame = DataFrame(np.arange(8).reshape((2, 4)),
                 index=['three', 'one'],
                 columns=['d', 'a', 'b', 'c'])
frame.sort_index()


# In[3]:


frame.sort_index(axis=1)


# In[4]:


#the data is sorted in ascending order by default, but can be sorted in descending order, too:
frame.sort_index(axis=1, ascending=False)


# In[6]:


#Any missing values are sorted to the end of the Series by default:
obj = Series([4, np.nan, 7, np.nan, -3, 2])
obj.sort_values()


# In[7]:


#When sorting a dataframe, you can use the data in one or more columns as the sort keys. To do so, pass one or more column
# names to the by option of sort_values:
frame = DataFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0, 1]})
frame


# In[8]:


frame.sort_values(by='b')


# In[9]:


frame.sort_values(by=['a', 'b'])


# In[2]:


#Ranking assigns ranks from one through the number of valid data points in an array. The rank methods for Series and Dataframe
# are the place to look; by default rank breaks ties by assigning each group the mean rank:
obj = Series([7, -5, 7, 4, 2, 0, 4])
obj.rank()


# In[3]:


#Ranks can also be assigned according to the order in which they're observed in the data: 
obj.rank(method='first')
#Here instead of using the average rank 6.5 for the entries 0 and 2 they instead have been set to 6 and 7


# In[4]:


#You can rank in descending order, too: 
#Assign tie values the maximum rank in the group
obj.rank(ascending=False, method='max')


# In[11]:


#DataFrame can compute ranks over the rows or the columns:
frame = DataFrame({'b': [4.3, 7, -3, 2], 
                   'a': [0, 1, 0, 1], 
                   'c': [-2, 5, 8, -2.5]})
frame.rank(axis='columns')


# In[12]:


#Axis Indexes with Duplicate Labels
#While many pandas functions (like reindex) require that the labels be unique, it's not mandatory:
obj = Series(range(5), index=['a', 'a', 'b', 'b', 'c'])
obj


# In[13]:


obj.index.is_unique


# In[17]:


#Data selection is one of the main things that behaves differently with duplicates. Indexing a label with multiple entries
# returns a Series, while single entries return a scalar value:
obj['a']


# In[15]:


obj['c']


# In[18]:


#this can make code complicated. Output type from indexing can vary based on whether a label is repeated or not
#THe same logic extends to indexing rows in a DataFrame:
df = DataFrame(np.random.randn(4, 3), index =['a', 'a', 'b', 'b'])
df


# In[19]:


df.loc['b']


# In[24]:


#Summarizing and Computing Descriptive Statistics
# pandas objects are equipped with a set of common mathematical and statistical methods. Most of these fall into the category
# of reductions or summary statistics, methods that extract a sengle value(like sum or mean) from a Series or DataFrame
df = DataFrame([[1.4, np.nan], 
              [7.1, -4.5],
              [np.nan, np.nan],
              [0.75, -1.3]],
              index=['a', 'b', 'c', 'd'],
              columns=['one', 'two'])
df


# In[26]:


#calling DataFrame's sum method returns a Series containing column sums:
df.sum()


# In[27]:


#passing axis='columns' or axis=1 sums across the columns instead:
df.sum(axis='columns')


# In[31]:


#NA values are excluded unless the entire slice (row or column in this case) is NA. This can be disabled with skipna option
df.mean(axis='columns', skipna=False)


# In[32]:


#Some methods like idxmin and idxmax return indirect statistics like the index value where the minimum or maximum values are attained
df.idxmax()


# In[33]:


#Other methods are accumulations:
df.cumsum()


# In[34]:


#Another type of method is neither a reduction nor an accumulation. describe is one such example:
df.describe()


# In[35]:


#On non-numerical data, describe produces alternative sumary statistics:
obj = Series(['a', 'a', 'b', 'c'] * 4)
obj.describe()


# In[36]:


#Correlation and Covariance
#Some summary statistics, like correlation and covariance, are computed from pairs of arguments. 
import pandas_datareader.data as web

all_data = {ticker: web.get_data_yahoo(ticker)
           for ticker in ['AAPL', 'IBM', 'MSFT', 'GOOG']}
price = DataFrame({ticker: data['Adj Close']
                  for ticker, data in all_data.items()})
volume = DataFrame({ticker: data['Volume']
                   for ticker, data in all_data.items()})
returns = price.pct_change()
returns.tail()


# In[38]:


#I now compute percent changes of the prices
returns = price.pct_change()
returns.tail()


# In[39]:


returns['MSFT'].corr(returns['IBM'])


# In[40]:


returns['MSFT'].cov(returns['IBM'])


# In[41]:


#Since MSFT is a valid Python attribute, we can also select these columns using more concise syntax:
returns.MSFT.corr(returns.IBM)


# In[42]:


#DataFrame's corr and cov methods, on the other hand, return a full correlation or covariance matrix as a DataFrame: 
returns.corr()


# In[43]:


returns.cov()


# In[44]:


#Using DataFrame's corrwith method, you can compute pairwise correlations between a DataFrame's columns or rows with 
# another Series or DataFrame. Passing a Series returns a Series with the correlation value computed for each column:
returns.corrwith(returns.IBM)


# In[45]:


#Passing a DataFrame computes the correlations of matching column names. Here I compute correlations of percent changes 
# with volume
returns.corrwith(volume)


# In[46]:


#Unique Values, Value Counts, and Membership
#Another class of related methods extracts information about the values contained in a one-dimensional Series.
obj = Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])
#The first function is unique, which gives you an array of the unique values in a Series:
uniques = obj.unique()
uniques


# In[47]:


#The unique values are not necessarily returned in sorted order, but could be sorted after the fact if needed (uniques.sort())
# Relatedly, value_counts computes a Series containing value frequencies:
obj.value_counts()


# In[48]:


#The Series is sorted by value in descending order as a convenience. value_counts is also available as a top-level pandas
# method that can be used with any array or sequence:
pd.value_counts(obj.values, sort=False)


# In[49]:


#isin performs a vectorized set membership check and can be useful in filtering a dataset down to a subset of values in a
# Series or column in a DataFrame:
obj


# In[51]:


mask = obj.isin(['b', 'c'])
mask


# In[53]:


obj[mask]


# In[4]:


#page 163
#related to isin is the Index.get_indexer method, which gives you an index array from an array of possibly non-distinct values
to_match = Series(['c', 'a', 'b', 'b', 'c', 'a'])
unique_vals = Series(['c', 'b', 'a'])
pd.Index(unique_vals).get_indexer(to_match)


# In[5]:


data = DataFrame({'Qu1':[1, 3, 4, 3, 4],
                 'Qu2': [2, 3, 1, 2, 3],
                 'Qu3': [1, 5, 2, 4, 4]})
data


# In[6]:


#passing pandas.value_counts to this DataFrame's apply function gives:
result = data.apply(pd.value_counts).fillna(0)
result


# In[ ]:


#here the row labels in the result are the distinct values ocurring in all of the columns. The values are the respective 
#counts of these values in each column. 

