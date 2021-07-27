#!/usr/bin/env python
# coding: utf-8

# In[24]:


class song:
    
    def __init__(self, lyrics):
        self.lyrics = lyrics
        
    def sing(self):
        for line in self.lyrics:
            print(line)
            
a_paris = song([""" À Paris
 Quand un amour fleurit
 Ça fait pendant des semaines
 Deux coeurs qui se sourient
 Tout ça parce qu'ils s'aiment"""])


a_paris.sing()


# In[ ]:




