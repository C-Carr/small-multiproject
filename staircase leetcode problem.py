#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# You have n coins and you want to build a staircase with these coins. 
# The staircase consists of k rows where the ith row has exactly i coins. The last row of the staircase may be incomplete.

# Given the integer n, return the number of complete rows of the staircase you will build.

class Solution:
    def arrangeCoins(self, n: int) -> int:
        my_list = [1]
        count = 0
        while n > sum(my_list):    
            my_list.append(len(my_list)+1)   

        for i in my_list:
            while n > 0:
                n -= my_list[-count]
                count += 1
        return count

