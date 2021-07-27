#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Accept Functions for Simple Interfaces Instead of Classes
from collections import defaultdict


names = ['Socrates', 'Archimedes', 'Plato', 'Aristotle']
names.sort(key=lambda x: len(x))
print(names)


# In[2]:


#Hook that logs each time a key is missing and returns ) for the default value
def log_missing():
    print('Key added')
    return 0

current = {'green': 12, 'blue': 3}
increments = [('red', 5),
             ('blue', 17),
             ('orange', 9),
             ]
result = defaultdict(log_missing, current)
print('Before:', dict(result))
for key, amount in increments:
    result[key] += amount
print('After: ', dict(result))


# In[3]:


#page 70 - Here I define a helper function that uses a stateful closure as the default value hook

def increment_with_report(current, increments):
    added_count = 0

    def missing():
        nonlocal added_count    #stateful closure
        added_count += 1
        return 0
    
    result = defaultdict(missing, current)
    for key, amount in increments:
        result[key] += amount
        
    return result, added_count


# In[4]:


result, count = increment_with_report(current, increments)
assert count == 2


# In[5]:


# ^might be hard to read. Another way is to define a small class that encapsulates the state you want to track

class CountMissing(object):
    def __init__(self):
        self.added = 0
        
    def missing(self):
        self.added += 1
        return 0


# In[6]:


counter = CountMissing()
result = defaultdict(counter.missing, current) #Method ref

for key, amount in increments:
    result[key] += amount
assert counter.added == 2


# In[7]:


# __call__ allows an object to be called just like a function

class BetterCountMissing(object):
    def __init__(self):
        self.added = 0
        
    def __call__(self):
        self.added += 1
        return 0
    
counter = BetterCountMissing()
counter()
assert callable(counter)


# In[8]:


counter = BetterCountMissing()
result = defaultdict(counter, current)  # Relies on __call__
for key, amount in increments:
    result[key] += amount
assert counter.added == 2


# In[9]:


#Page 72
# Use @classmethod Polymorphism to Construct Objects Generically
# writing a MapReduce implementation, want a common class to represent the input data. Define such a class with a read
# method that must be defined by subclasses

class InputData(object): 
    def read(self):
        raise NotImplementedError
        
class PathInputData(InputData):       # concrete subclass of InputData that reads data from a file on disk
    def __init__(self, path):
        super().__init__()
        self.path = path
        
    def read(self):
        return open(self.path).read()
    
class Worker(object):                # similar abstract interface for the MapReduce worker that consumes the input data in a standard way
    def __init__(self, input_data):
        self.input_data = input_data
        self.result = None
        
    def map(self):
        raise NotImplementedError
        
    def reduce(self, other):
        raise NotImplementedError


# In[10]:


#what connects? simplest approach is to manually build and conext objects with heper functions
#list the contents of a directory and construct a PathInputData instance for each file it contains

import os
import random

def generate_inputs(data_dir):
    for name in os.listdir(data_dir):
        yield PathInputData(os.path.join(data_dir, name))
        
#create LineCountWorker instances using the InputData instances returned by generate_inputs
def create_workers(input_list):
    workers = []
    for input_data in input_list:
        workers.append(LineCounterWorker(input_data))
    return workers

# Execute Worker instances by fanning out the map step into multiple throeads then call reduce repeatedly to combine results into one final value
def execute(workers):
    threads = [Thread(target=w.map) for w in workers]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
        
    first, rest = workers [:], workers[1:]
    for worker in rest:
        first.reduce(worker)
    return first.result

#Connect all of the pieces together in a function to run each step

def mapreduce(data_dir):
    inputs = generate_inputs(data_dir)
    workers = create_workers(inputs)
    return execute(workers)

from tempfile import TemporaryDirectory
import tempfile

def write_test_files(tmpdir):
    tmpdir = tempfile.TemporaryFile()
    tmpdir.write(b'Hello world! \n'*100)    
    
with TemporaryDirectory() as tmpdir:
    write_test_files(tmpdir)
    result = mapreduce(tmpdir)
    
print(f'There are {result} lines')
        
#huge issue is that mapreduce function is not generic. If you want to write another Inputdata or Worker subclass 
# you would have to rewrite the generate_inputs, create_workers and mapreduce funcitons to match


# In[11]:


#python doesn't have a special constructor that can used generically so use classmethod to make new InputData generic classes

class GenericInputData(object):
    def read(self):
        raise NotImplementedError
        
    @classmethod
    def generate_inputs(cls, config):
        raise NotImplementedError
        
class PathInputData(GenericInputData):
    # ...
    def read(self):
        return open(self.path).read()
    
@classmethod
def generate_inputs(cls, config):
    data_dir = config['data_dir']
    for name in os.listdir(data_dir):
        yield cls(os.path.join(data_dir, name))
        
class GenericWorker(object):
    #...
    def map(self):
        raise NotImplementedError
        
    def reduce(self, other):
        raise NotImplementedError
        
    @classmethod
    def create_workers(cls, input_class, config):
        workers = []
        for input_data in input_class.generate_inputs(config):
            workers.append(cls(input_data))
        return workers


# In[13]:


import os
help(os.mkdir)

