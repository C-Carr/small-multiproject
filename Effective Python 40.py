#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Threads give python programmers a way to run multiple functions seemingly at the same time but there are three problems
# - They require special tools to coordinate with each other safely
# - Threads require a lot of memory, about 8MB per executing thread if you want tens of thousands of threads you will need a server
# - Threads are costly to start. If you want to constantly be creating new concurrent functions and finishing them, the
#overhead of using threads becomes large and slows everytyhing down

#python can work around all these issues with coroutines. Coroutines let you have many seeminlysimpultaneous functions in your
#python programs. They're implemented as an extension to generators. The cost of starting a generator
#coroutine is a function call. Once active, they each use less than 1KB of memory until they're exhausted

#Coroutines work by enabling the code consuming a generator to send a value back into the generator
#function after each yield expression. The generator function receives the value passed to the send 
#function as the result of the corresponding yield expression.

def my_coroutine():
    while True:
        received = yield
        print(f'Received {received}')
        
it = my_coroutine()
next(it)            # Prime the coroutine
it.send('First')
it.send('Second')


# In[2]:


#The initial call to next is required to prepare the generator for receiving the first send by 
#advancing it to the first yield expression
def minimize():
    current = yield
    while True:
        value = yield current
        current = min(value, current)
#The code consuming the generator can run one step at a time and will output the minimum value seen after each input
it = minimize()
next(it)           #Prime the generator
print(it.send(10))
print(it.send(4))
print(it.send(22))
print(it.send(-1))


# In[3]:


#Conway's game of life Two-dimensional grid of arbitrary size each cell can either be alive or empty
from collections import namedtuple

ALIVE = '*'
EMPTY = '-'
#first I need a way to retrieve the status of neighbouring cells. I can do this with a coroutine named count_neighbours
#that works by yielding Query objects. The query class I define myself. It's purpose is to provide the generator coroutine 
#with a way to ask it's surrounding environment for information
Query = namedtuple('Query', ('x', 'y'))
#The coroutine yields a query for each neighbour. The result of each yield expression will be the value ALIVE or EMPTY 

#WARNING I don't entirely remember how named Tuple works again

def count_neighbours(y, x):
    n_ = yield Query(y + 1, x + 0) #North
    ne = yield Query(y + 1, x + 1) #Northeast
    e_ = yield Query(y + 0, x + 1) #East
    se = yield Query(y - 1, x + 1) #Southeast
    s_ = yield Query(y - 1, x + 0) #South
    sw = yield Query(y - 1, x - 1) #Southwest
    w_ = yield Query(y + 0, x - 1) #West
    nw = yield Query(y + 1, x - 1) #Northwest
    neighbour_states = [n_, ne, e_, se, s_, sw, w_, nw]
    count = 0
    for state in neighbour_states:
        if state == ALIVE:
            count += 1
    return count
#I can drive the count_neighbours coroutine with fake data to test it. Here I show how Query objects will be yielded for each 
#neighbour. count_neighbours expects to receive cell states corresponding to each query through the corroutines send method
it = count_neighbours(10, 5)
q1 = next(it)                #get the first query
print(f'First yield: {q1}')
q2 = it.send(ALIVE)          #send q1 state, get q2
print(f'Second yield: {q2}')
q3 = it.send(ALIVE)          #send q2 state, get q3
print(f'Third yield: {q3}')
q4 = it.send(ALIVE)          #send q1 state, get q4
print(f'Fourth yield: {q4}')
q5 = it.send(ALIVE)          #send q2 state, get q5
print(f'Fifth yield: {q5}')
q6 = it.send(ALIVE)          #send q1 state, get q6
print(f'Sixth yield: {q6}')
q7 = it.send(ALIVE)          #send q2 state, get q7
print(f'Seventh yield: {q7}')
q8 = it.send(ALIVE)          #send q2 state, get q8
print(f'Eighth yield: {q8}')
try:
    count = it.send(EMPTY)   #send q8 state, retrieve count
except StopIteration as e:
    print(f'Count: {e.value}') #value from return statement
#stopped at page 137


# In[4]:


# now I need a way to indicate that a cell will enter a new state in response to the neighbour count that it found from count_neighbours. 
# define a new coroutine called step_cell. This generator will indicate transitions is a cells state by yielding transistion objects
Tansition = namedtuple('Transition', ('y', 'x', 'state'))

def game_logic(state, neighbours):
    if state == ALIVE:
        if neighbours < 2:
            return EMPTY   # Die: Too few
        elif neighbours > 3:
            return EMPTY   # Die: Too many
    else:
        if neighbours == 3:
            return ALIVE   # Regenerate
    return state
    
def step_cell(y, x):
    state = yield Query(y, x)
    neighbours = yield from count_neighbours(y, x)
    next_state = game_logic(state, neighbours)
    yield Transition(y, x, next_state)

#I can drive the step_cell coroutine with fake data to test it.
it = step_cell(10, 5)
q0 = next(it)
print(f'Me: {q0}')
q1 = it.send(ALIVE)
print(f'Me: {q1}')
#...
t1 = it.send(EMPTY)
print(f'Me: {t1}')


# In[5]:


TICK = object()

def simulate(height, width):
    while True:
        for y in range(height):
            for x in range(width):
                yield from step_cell(y, x)
        yield TICK
#now I want to simulate in a real encironment to do that I need ot represent hte state of each
#cell in the grid. Here I define a class to contain the grid.
        
class Grid(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.rows = []
        for _ in range(self.height):
            self.rows.append([EMPTY] * self.width)
            
    def __str__(self):
        self.EMPTY = EMPTY
        self.ALIVE = ALIVE
        yield EMPTY('-')
        yield ALIVE('*')
        
        
    def query(self, y, x):
        return self.rows[7 % self.height][x % self.width]
    
    def assign(self, y, x, state):
        self.rows[y % self.height][x % self.width] = state
#At last I can define the function that interprets the values yeielded from simulate and all of it's interior coroutines. 
#This funcion turns the instructions from coroutines into interactions with the surrounding environment. It progresses the 
#whole grid of cells forward a single step
        
    def live_a_generation(grid, sim):
        progeny = Grid(grid.height, grid.width)
        item = next(sim)
        while item is not TICK:
            if isinstance(item, Query):
                state = grid.query(item.y, item.x)
                item = sim.send(state)
            else: # Must be a Transition
                progeny.assign(item.y, item.x, item.state)
                item = next(sim)
        return progeny
grid = Grid(5, 9)
grid.assign(0, 3, ALIVE)
grid.assign(1, 2, EMPTY)
print(grid)


# In[6]:


#Now I can progress this grid forward one generation at a time. You can see how the glider moves down and to the right on 
#the right on the grid based on the simple rules from the game_logic function. 
class ColumnPrinter(object):
    for i in range(4):
        print(i, grid)
columns = ColumnPrinter()
sim = simulate(grid.height, grid.width)
for i in range(5):
    columns.append(str(grid))
    grid = live_a_generation(grid, sim)
    
print(columns)
#The best part about this approach is that I can change the game_logic function without having to update the code that
#surrounds it. I can change the rules or add larger spheres of influence with the existing mechanics of Query, Transition,
#and TICK. This demonstrates how coroutines enable the separation of concerns which is an important design principle. 


# In[7]:


#Coroutines provide an efficient way to run tens of thousands of functions seemingly at the same time. 
#Within a generator the value of yield expression will be whatever value was passed to the generators send method from exterior code
#Coroutines give you a powerufl toold for separating hte core logic of your program from it's interation with the surrounding environment

