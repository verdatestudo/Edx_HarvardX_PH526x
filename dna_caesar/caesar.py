'''
Week 3/4 - Case Study 1
from HarvardX: PH526x Using Python for Research on edX

A cipher is a secret code for a language. In this case study, we will explore a cipher that is reported by
contemporary Greek historians to have been used by Julius Caesar to send secret messages to generals during times of war.

Last Updated: 2016-Dec-20
First Created: 2016-Dec-20
Python 3.5
Chris
'''

# Let's look at the lowercase letters.
import string

def caesar(message, key):
    '''
    Takes a message string and an int key and returns the coded caesar message.
    '''
    # define `coded_message` here!
    coded_message = {letter: (idx + key) % 27 for idx, letter in enumerate(alphabet)}
    return ''.join([letters[coded_message[letter]] for letter in message])

# We will consider the alphabet to be these letters, along with a space.
alphabet = string.ascii_lowercase + " "

# create `letters` here!
letters = {idx: letter for idx, letter in enumerate(alphabet)}

# Use caesar to encode message using key = 3, and save the result as coded_message.
message = "hi my name is caesar"
coded_message = caesar(message, 3)
print(coded_message)
decoded_message = caesar(coded_message, -3)
print(decoded_message)
