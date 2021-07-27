#!/usr/bin/env python
# coding: utf-8

# In[17]:


# create a mapping of state to abbreviation
provinces = {
    'British Columbia': 'BC',
    'Alberta': 'AB',
    'Saskatchewan': 'SK',
    'Ontario': 'ON',
    'Quebec': 'QB'
}

# create a basic set of provinces and some cities in them
cities = {
    'BC': 'Vancouver',
    'QB': 'Montreal',
    'SK': 'Regina'
}

# add some more cities
cities['AB'] = 'Calgary'
cities['ON'] = "Toronto"

# print out some cities
print('-' * 10)
print("ON Province has: ", cities['ON'])
print("SK Province has: ", cities['SK'])

#print some provinces
print('-' * 10)
print("Quebec's abbreviation is: ", provinces['Quebec'])
print("British Columbia's abbreviation is: ", provinces['British Columbia'])

# do it by using the province then cities dict
print('-' * 10)
print("Quebec has: ", cities[provinces['Quebec']])
print("British Columbia has: ", cities[provinces['British Columbia']])

# print every province abbreviation
print('-' * 10)
for province, abbrev in list(cities.items()):
    print(f"{province} is appreviated {abbrev}")
    
# print every city in the state
print('-' * 10)
for abbrev, city in list(provinces.items()):
    print(f"{abbrev} has the city {city}")
    
# now do both at the same time
print('-' * 10)
for province, abbrev in list(provinces.items()):
    print(f"{province} province is abbreviated {abbrev}")
    print(f"and has city {cities[abbrev]}")
    
print('-' * 10)
#safely get an abbreviation by province that might not be there
province = provinces.get('Manitoba')

if not province:
    print("Sorry, no Manitoba.")
    
# get a city with a defoult value
city = cities.get('MB', 'Does Not Exist')
print(f"The city for the province 'MB' is: {city}")


# In[ ]:




