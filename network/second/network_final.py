'''
Week 3/4 - Case Study 6
from HarvardX: PH526x Using Python for Research on edX

Network homophily occurs when nodes that share an edge share a characteristic more often than nodes that do not share an edge.
In this case study, we will investigate homophily of several characteristics of individuals connected in social networks in rural India.

Last Updated: 2016-Dec-25
First Created: 2016-Dec-24
Python 3.5
Chris

# https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/
'''

###
###


# individual_characteristics.dta contains several characteristics for each individual in the dataset such as age, religion, and caste.
# Use the pandas library to read in and store these characteristics as a dataframe called df.
# Store separate datasets for individuals belonging to Villages 1 and 2 as df1 and df2, respectively.
# (Note that some attributes may be missing for some individuals. Here, investigate only those pairs of nodes where the attributes are known for both nodes.
# This means that we're effectively assuming that the data are missing completely at random.)
# Use the head method to display the first few entries of df1.

import pandas as pd
from collections import Counter
import numpy as np

df  = pd.read_stata("individual_characteristics.dta")

df1 = df[df.village == 1]
df2 = df[df.village == 2]

# print(df1.head())

###
###

# In this dataset, each individual has a personal ID, or PID, stored in key_vilno_1.csv and key_vilno_2.csv for villages 1 and 2, respectively.
# data_filepath contains the base URL to the datasets used in this exercise. Use pd.read_csv to read in and store key_vilno_1.csv and key_vilno_2.csv as pid1 and pid2 respectively.
# The csv files have no headers, so make sure to include the parameter header = None.

pid1 = pd.read_csv('key_vilno_1.csv', header=None)
pid2 = pd.read_csv('key_vilno_2.csv', header=None)

###
###

# Define Python dictionaries with personal IDs as keys and a given covariate for that individual as values.
# Complete this for the sex, caste, and religion covariates, for Villages 1 and 2.
# Store these into variables named sex1, caste1, and religion1 for Village 1 and sex2, caste2, and religion2 for Village 2.

# 'characteristic', try df1.set_index("pid")[characteristic], and use .to_dict() to convert this object to a dict.

# print(df1.columns)

sex1 = df1.set_index('pid')['resp_gend'].to_dict()
sex2 = df2.set_index('pid')['resp_gend'].to_dict()
caste1 = df1.set_index('pid')['caste'].to_dict()
caste2 = df2.set_index('pid')['caste'].to_dict()
religion1 = df1.set_index('pid')['religion'].to_dict()
religion2 = df2.set_index('pid')['religion'].to_dict()

###
###

# Let's consider how much homophily exists in these networks. For a given characteristic, our measure of homophily will be the proportion of edges in the network whose constituent nodes share that characteristic.
# How much homophily do we expect by chance? If characteristics are distributed completely randomly, the probability that two nodes share a characteristic is simply the product of chances that each node independently has that characteristic.
# To find the probability they share a given characteristic, we then simply sum the chances of sharing that characteristic. How can we do this for our dataset?
# Create a function chance_homophily(chars) that takes a dictionary with personal IDs as keys and characteristics as values, and computes the chance homophily for that characteristic.
# A sample of three peoples' favorite colors is given in favorite_colors. Use your function to compute the chance homophily in this group.

# the probability that two nodes share a characteristic is simply the PRODUCT of chances that each node independently has that characteristic.
# To find the probability they share a given characteristic, we then simply sum the chances of sharing that characteristic. How can we do this for our dataset?
# Create a function chance_homophily(chars) that takes a dictionary with personal IDs as keys and characteristics as values, and computes the chance homophily for that characteristic.
# A sample of three peoples' favorite colors is given in favorite_colors. Use your function to compute the chance homophily in this group.

def chance_homophily(chars):
    """
    Computes the chance homophily of a characteristic,
    specified as a dictionary, chars.
    """
    chars_counts_dict = Counter(chars.values())
    chars_counts = np.array(list(chars_counts_dict.values()))
    chars_props  = chars_counts / sum(chars_counts)
    return sum(chars_props**2)

favorite_colors = {
    "ankit":  "red",
    "xiaoyu": "blue",
    "mary":   "blue"
}

chance_homophily(favorite_colors)

###
###

# sex1, caste1, religion1, sex2, caste2, and religion2 are already defined from previous exercises.
# Use chance_homophily to compute the chance homophily for sex, caste, and religion In Villages 1 and 2.
# Is the chance homophily for any attribute very high for either village?

print(chance_homophily(sex1))
print(chance_homophily(sex2))
print(chance_homophily(caste1))
print(chance_homophily(caste2))
print(chance_homophily(religion1))
print(chance_homophily(religion2))

###
###

# Now let's compute the observed homophily in our network. Recall that our measure of homophily is the proportion of edges whose nodes share a characteristic.
# homophily(G, chars, IDs) takes a network G, a dictionary of characteristics chars, and node IDs IDs. For each node pair, determine whether a tie exists between them,
# as well as whether they share a characteristic. The total count of these is num_same_ties and num_ties respectively, and their ratio is the homophily of chars in G.
# Complete the function by choosing where to increment num_same_ties and num_ties.

def homophily(G, chars, IDs):
    """
    Given a network G, a dict of characteristics chars for node IDs,
    and dict of node IDs for each node in the network,
    find the homophily of the network.
    """
    num_same_ties, num_ties = 0, 0
    for n1 in G.nodes():
        for n2 in G.nodes():
            if n1 > n2:   # do not double-count edges!
                if IDs[n1] in chars and IDs[n2] in chars:
                    if G.has_edge(n1, n2):
                        num_ties += 1
                        if chars[IDs[n1]] == chars[IDs[n2]]:
                            num_same_ties += 1
    return (num_same_ties / num_ties)

###
###

# The networks for Villages 1 and 2 have been stored as networkx graph objects G1 and G2. (df1 and df2 on local machine)
# Use your homophily function to compute the observed homophily for sex, caste, and religion in Villages 1 and 2.
# Print all six values. Are these values higher or lower than that expected by chance?

print("Village 1 observed proportion of same sex:", homophily(G1, sex1, pid1))
print("Village 2 observed proportion of same sex:", homophily(G2, sex2, pid2))
print("Village 1 observed proportion of same caste:", homophily(G1, caste1, pid1))
print("Village 2 observed proportion of same caste:", homophily(G2, caste2, pid2))
print("Village 1 observed proportion of same religion:", homophily(G1, religion1, pid1))
print("Village 2 observed proportion of same religion:", homophily(G2, religion2, pid2))

###
###