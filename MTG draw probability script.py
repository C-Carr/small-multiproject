#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[ ]:




