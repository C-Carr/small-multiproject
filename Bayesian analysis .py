#!/usr/bin/env python
# coding: utf-8

# In[25]:


#P(A|B) = P(B|A)xP(A)/P(B)
#P(A|B) = AB
#P(B|A) = BA
#P(A) = A
#P(B) = B

# my guess is I'm 90% likely to have a FP over TP
# I was super wrong. It's 42.9%.

# Looking for AB probability of a false positive antigen covid test given the positivity rate in my area
# A General positivity rate in my area = (1%)
# B Tests that come back positive 2.3%
# BA how many true positive results are there among positive cases (98.7%)
# P(test=positive)=P(test=positive|covid positive)*P(covidpositive)+P(test=positive|Covid negative)*P(Covid negative)

A = .01
BA = .987
B = .023

AB = ((BA*A)/B)*100
print(f'{"{:.2f}".format(AB)}%')

