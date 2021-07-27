#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Register Class Existence with Metaclasses
#another common use of metaclasses is to automatically register types in your program
#Registration is useful for doing reverse lookups where you need to map a simple identifier back to corresponding class
#this takes an object and turns it into a JSON sting. this is done generically by defining a base class that records the constructor parameters and turns them into a JSON dict
import json
class Serializable(object):
    def __init__(self, *args):
        self.args = args
        
    def serialize(self):
        return json.dumps({'args': self.args})
#this class makes it easy to serialize, immutable structures like Point2D to a string
class Point2D(Serializable):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.x = x
        self.y = y
        
    def __repr__(self):
        return f'Point2D{self.x, self.y}'
    
point = Point2D(5, 3)
print(f'Object: {point}')
print(f'Serialized: {point.serialize()}')


# In[2]:


class Deserializable(Serializable):
    @classmethod
    def deserialize(cls, json_data):
        params = json.loads(json_data)
        return cls(*params['args'])
#using deserializable makes it easy to serialize and deserialize simple immutable objects in a generic way
class BetterPoint2D(Deserializable):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.x = x
        self.y = y
        
    def __repr__(self):
        return f'BetterPoint2D{self.x, self.y}'
    
point = BetterPoint2D(5, 3)
print(f'Before: {point}')
data = point.serialize()
print(f'Serialized: {data}')
after = BetterPoint2D.deserialize(data)
print(f'After: {after}')
#This only works if you know the intended type of the serialized data ahead of time
#Ideally you'd have a large number of classes serializing to JSON and one common function that could
#deserialize any of them backto a corresponding object


# In[3]:


#to do this you can include the serialized object's class name in the JSON data
class BetterSerializable(object):
    def __init__(self, *args):
        self.args = args
        
    def serialize(self):
        return json.dumps({
            'class': self.__class__.__name__,
            'args': self.args,
        })
    def __repr__(self):
        return BetterSerializable(self)

class EvenBetterPoint2D(BetterSerializable):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.x = x
        self.y = y
    def __repr__(self):
        return(f'EvenBetterPoint2D{self.x, self.y}')

#then I can maintain a mapping of class names back to constructors for those objects
#the general deserialize function will work for any classes passed to register_class
registry = {} 
def register_class(target_class): 
    registry[target_class.__name__] = target_class
        

def deserialize(data):
    params = json.loads(data)
    name = params['class']
    target_class = registry[name]
    return target_class(*params['args']) 
    
 
register_class(EvenBetterPoint2D)
#to ensure that deserialize works properly I have to call register_class for every class I may want to deserialize in the future
#now I can deserialze an arbitrary JSON string without having to know which class it contains
point = EvenBetterPoint2D(5, 3)
print(f'Before: {point}')
data = point.serialize()
print(f'Serialized: {data}')
after = deserialize(data)
print(f'After: {after}')


# In[4]:


#the problem with this is you can forget to call register_class
class Point3D(BetterSerializable):
    def __init__(self, x, y, z):
        super().__init__(x, y, z)
        self.x = x
        self.y = y
        self.z = z    
    def __repr__(self):
        return f'Point3D{self.x, self.y, self.z}'
            
point = Point3D(5, 9, -4)
data = point.serialize()
deserialize(data)
#even though you sublcassed BetterSerializable you won't get all of it's features if you forget to call register_class after your class statement body


# In[5]:


class Meta(type):
    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        register_class(cls)
        return cls
    
class RegisteredSerializable(BetterSerializable,
                          metaclass=Meta):
    pass

class Vector3D(RegisteredSerializable):
    def __init__(self, x, y, z):
        super().__init__(x, y, z)
        self.x, self.y, self.z = x, y, z
    def __repr__(self):
        return f'Vector3D{self.x, self.y, self.z}'

v3 = Vector3D(10, -7, 3)
print(f'Before: {v3}')
data = v3.serialize()
print(f'Serialized: {data}')
print(f'After: {deserialize(data)}')


# In[ ]:


#Class registration is a helpful pattern for building modular python programs
#Metaclasses let you run registration code automatically each time your base class is subclassed in a program
#using metaclasses for class registration avoids errors by ensuring that you never miss a registration call

