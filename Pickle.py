#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Python offers three different modules in the standard library that allow you to serialize and deserialize objects
# The marshal module, the json module, the pickle module. Python also supports XML which you can use to serialize objects
# with Json you can serialize several standard Python types. Bool, dict, int, float, list, string, tuple, and None
# The python pickle module basically consists of four methods
# pickle.dump(obj, file, protocol=None, *, fix_imports=True, buffer_callback=None) - creates a file
# pickle.dumps(obj, protocol=None, *, fix_imports=True, buffer_callback=None)      - returns a string
# pickle.load(file, *, fix_imports=True, buffer_callback=None)                                     -file
# pickle.loads(bytes_object, *, fix_imports=True, encoding="ASCII", errors="strict", buffers=None) - string

#pickling.py
import pickle

class example_class():
    a_square = lambda x : x * x
    a_string = "hey"
    a_list = [1, 2, 3]
    a_dict = {"first": "a", "second": 2, "third": [1, 2, 3]}
    a_tuple = (22, 23)
    
my_object = example_class()

my_pickled_object = pickle.dumps(my_object)  #pickling the object
print(
f"This is a_dict of the unpickled object:\n{my_pickled_object}")

my_object.a_dict = None

square = example_class.a_square

my_unpickled_object = pickle.loads(my_pickled_object) #unpickling the object
print(
f"This is a_square(2) of the unpickled object:\n{square(2)}\n")

import os
import pickle

def read_or_new_pickle(path, default=3):
    if os.path.isfile(path):
        with open(path, "rb") as f:
            try:
                return pickle.load(f)
            except Exception: # so many things could go wrong, can't be more specific.
                pass 
    with open(path, "wb") as f:
        pickle.dump(default, f)
    return default


# In[2]:


class example_class():
    a_square = lambda x : x * x
    a_string = "hey"
    a_list = [1, 2, 3]
    a_dict = {"first": "a", "second": 2, "third": [1, 2, 3]}
    a_tuple = (22, 23)

square = example_class.a_square
square(2)


# In[3]:


my_object.a_dict


# In[4]:


my_unpickled_object.a_dict


# In[5]:


# the dill module extends the capabilities of pickle. According to the official documentation if lets you serialize less common 
#types like funtions with yields, nested functions, lambdas, and many others.
#pickling_error.py
import pickle

square = lambda x : x * x
my_pickle = pickle.dumps(square)


# In[6]:


class my_square():
    square = lambda x : x * x
    
my_pickle = pickle.dumps(my_square.square(6))


# In[7]:


my_unpickle = pickle.loads(my_pickle)
my_unpickle


# In[8]:


square = lambda x : x * x

my_pickle = pickle.dumps(square(3))


# In[9]:


my_unpickle = pickle.loads(my_pickle)
my_unpickle


# In[10]:


my_pickle = pickle.dumps(square)


# In[11]:


import dill

square = lambda x : x * x
a = square(35)
import math
b = math.sqrt(484)
import dill
dill.dump_session('test.pkl')
exit()


# In[12]:


# the solution for database connections is to exclude the object from the serialization process and to reinitialize the connection after the object is deserialized
import pickle

class foobar: 
    def __init__(self):
        self.a = 35
        self.b = "test"
        self.c = lambda x: x * x
        
    def __getstate__(self):
        attributes = self.__dict__.copy()
        del attributes['c']
        return attributes

my_foobar_instance = foobar()
my_pickle_string = pickle.dumps(my_foobar_instance)
my_new_instance = pickle.loads(my_pickle_string)

print(my_new_instance.__dict__)


# In[13]:


import pickle

class foobar: 
    def __init__(self):
        self.a = 35
        self.b = "test"
        self.c = lambda x: x * x
        
    def __getstate__(self):
        attributes = self.__dict__.copy()
        del attributes['c']
        return attributes
    
    def __setstate__(self, state):
        self.__dict__ = state
        self.c = lambda x: x * x
        
my_foobar_instance = foobar()
my_pickle_string = pickle.dumps(my_foobar_instance)
my_new_instance = pickle.loads(my_pickle_string)

print(my_new_instance.__dict__)


# In[14]:


import pickle
import bz2

my_string = """Per me si va ne la città dolente,
per me si va ne l'etterno dolore,
per me si va tra la perduta gente.
Giustizia mosse il mio alto fattore:
fecemi la divina podestate,
la somma sapienza e 'l primo amore;
dinanzi a me non fuor cose create
se non etterne, e io etterno duro.
Lasciate ogne speranza, voi ch'intrate."""
pickled = pickle.dumps(my_string)
compressed = bz2.compress(pickled)
len(my_string)


# In[15]:


len(compressed)


# In[16]:


dir(len)


# In[17]:


help(len)


# In[18]:


# Modify the variables so that all of the statements evaluate to True.

var1 = 34
var2 = ("pkwmin")
var3 = [1, 2, 3, 4, 5]
var4 = ("Hello", "Python!", "Hello, Python!")
var5 = {"egg" : "salad", "tuna" : "fish", "happy" : 7}
var6 = 36.0

# Don't edit anything below this comment

# Numbers
print(isinstance(var1, int))
print(isinstance(var6, float))
print(var1 < 35)
print(var1 <= var6)

# Strings
print(isinstance(var2, str))
print(var2[5] == "n" and var2[0] == "p")

# Lists
print(isinstance(var3, list))
print(len(var3) == 5)

# Tuples
print(isinstance(var4, tuple))
print(var4[2] == "Hello, Python!")

# Dictionaries
print(isinstance(var5, dict))
print("happy" in var5)
print(7 in var5.values())
print(var5.get("egg") == "salad")
print(len(var5) == 3)
var5["tuna"] = "fish"
print(len(var5) == 3)


# In[19]:


class car:   
    def __init__(self, colour, make_model, miles):
        self.colour = colour
        self.make_model = make_model
        self.miles = miles
        
    def description(self, sound):
        return f"The {self.colour} {self.make_model} has {self.miles:,} miles. It goes {sound}"
    
my_car = car(colour="blue", make_model="chevy_cavalier", miles=20_000)
your_car = car(colour="red", make_model="toyota_corola", miles=30_000)
gregs_car = car(colour="green", make_model="toyota_sienna", miles=40_000)

for car in (my_car, your_car, gregs_car):
    print(f"The {car.colour} {car.make_model} has {car.miles:,} miles.")


# In[20]:


my_car.description("vroom")


# In[21]:


class Car:
    def __init__(self, colour, mileage):
        self.colour = colour
        self.mileage = mileage
        
blue_car = Car(colour="blue", mileage=20_000)
red_car = Car(colour="red", mileage=30_000)

for car in (blue_car, red_car):
    print(f"The {car.colour} car has {car.mileage:,} miles")


# In[22]:


class Dog:
    species = "Canis familiaris"

    def __init__(self, name, age):
        self.name = name
        self.age = age
        
    def __str__(self):
        return f"{self.name} is {self.age} years old"

    def speak(self, sound):
        return f"{self.name} says {sound}"


# In[23]:


miles = Dog("Miles", 4, "Jack Russell Terrier")
buddy = Dog("Buddy", 9, "Dachsund")
jack = Dog("Jack", 3, "Bulldog")
jim = Dog("Jim", 5, "Bulldog")

buddy.speak("Yap")


# In[24]:


jim.speak("Woof")


# In[25]:


jack.speak("Woof")


# In[26]:


class JackRussellTerrier(Dog):
    def speak(self, sound="Arf"):
        return f"{self.name} says {sound}"

class Dachsund(Dog):
    def speak(self, sound="Grr"):
        return f"{self.name} says {sound}"

class Bulldog(Dog):
    def speak(self, sound="Bork"):
        return f"{self.name} says {sound}"

miles = JackRussellTerrier("Miles", 4)
buddy = Dachsund("Buddy", 9)
jack = Bulldog("Jack", 3)
jim = Bulldog("Jim", 5)

miles.species


# In[27]:


buddy.name


# In[28]:


print(jack)


# In[29]:


jim.speak("Woof")


# In[30]:


type(miles)


# In[31]:


isinstance(miles, Dog)


# In[32]:


miles = JackRussellTerrier("Miles", 4)
miles.speak()


# In[57]:


jim.speak()


# In[58]:


class JackRussellTerrier(Dog):
    def speak(self, sound="Arf"):
        return super().speak(sound)
    
miles = JackRussellTerrier("Miles", 4)
miles.speak()


# In[ ]:




