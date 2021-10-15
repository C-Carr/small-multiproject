#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random

def new_hand(num_lands, deck_size=60, hand_size=7):
    """
    Define size of deck and size of hand 
    Returns random hand of lands and spells
    """
    deck = []
    deck += ['L'] * num_lands + ['S'] * (deck_size - num_lands)
    return random.sample(deck, hand_size)

def count_lands_per_hand(practice_rounds, num_lands):
    """
    Counts number of 'lands' in hand 
    Returns land count
    """
    land_counts = {i: 0 for i in range(8)}
    for hand_num in range(practice_rounds):
        hand = new_hand(num_lands)
        hand_lands = hand.count('L')
        land_counts[hand_lands] += 1
    return land_counts
"""
convert to classes for more versatility in the future
Define number of lands, practice rounds, print results
"""
practice_rounds = 10000
num_lands = 20

land_counts = count_lands_per_hand(practice_rounds, num_lands)

for hand_lands, num_hands in sorted(land_counts.items()):
    percent_of_total = int((num_hands / practice_rounds) * 100)
    print_line = f'{hand_lands}-Land {num_hands:>2} ({percent_of_total:>2}%)'
    print_line = f'{hand_lands}-Land  {num_hands:>5}  ({(num_hands / practice_rounds):>6.2%})'

    print(print_line)


# In[4]:


import random
from pandas import DataFrame
"""
dictionary tree containing card-types as keys and number of cards as values main branches being lands and spells
Note: I can change the card types to denote important cards and change functionality to reflect chances of drawing combo
"""
deck = {'lands':{'green':'g' * 4,
              'blue': 'b' * 16,
              'red': 'r' * 0,
              'white': 'w' * 0,
              'black': 'l' * 0},
       'spells':{'splash': 's' * 3,
              'main': 'm' * 37}}

practice_rounds = 100000

def shuffle_deck(): 
    """
    Returns shuffled list of keys 
    """
    make_deck = []
    while len(make_deck) <= 60:
        for cards, types in deck.items():
            try:
                for key in types:
                    make_deck += types[key]
            except IndexError:
                pass
        random.shuffle(make_deck)
        return make_deck
       
def new_hand(shuffle_deck):
    """
    returns random 7 values from shuffle_deck
    """
    hand_size = 7
    hand = []
    for i in range(0, hand_size):
        hand.append(shuffle_deck[i])
    return hand

def count_lands():
    """
    returns count of cards in hand 
    could be made better by making a for loop and 'deck'
    for key in deck, lands,:
        draw.count(value)
    """
    find_green = {i: 0 for i in range(8)}
    find_blue = {i: 0 for i in range(8)}
    find_red = {i: 0 for i in range(8)}
    find_white = {i: 0 for i in range(8)}
    find_black = {i: 0 for i in range(8)}
    find_splash = {i: 0 for i in range(8)}
    find_main = {i: 0 for i in range(8)}
    for hand_num in range(practice_rounds):
        draw = new_hand(shuffle_deck())
        grn = draw.count('g')
        blu = draw.count('b')
        rd = draw.count('r')
        wht = draw.count('w')
        bl = draw.count('l')
        spl = draw.count('s')
        mn = draw.count('m')
        find_green[grn] += 1
        find_blue[blu] += 1
        find_red[rd] += 1
        find_white[wht] += 1
        find_black[bl] += 1
        find_splash[spl] += 1
        find_main[mn] += 1
    return find_green, find_blue, find_red, find_white, find_black, find_splash, find_main

def Deck_Frame():
    """
    Data from count_lands go into a DataFrame format with columns = card count 0-7, index = card types
    """
    hand_size = range(8)
    Frame_Count = DataFrame(count_lands(), 
                            index = ('green', 
                                    'blue',
                                    'red',
                                    'white',
                                    'black',
                                    'splash',
                                    'main'),
                           columns = (hand_size))
    return Frame_Count

def Find_Percent():
    """
    Apply percentage calculation across rows to calculate percentages.
    """
    percent = Deck_Frame().apply((lambda x: 100 * x / float(x.sum())), axis=1)
    return percent

Find_Percent()

