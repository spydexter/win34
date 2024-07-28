import numpy as np
import pandas as pd

# Loading Data from a CSV File
data = pd.read_csv('training-data.csv')
print(data)

# Separating concept features from Target
concepts = data.iloc[:, :-1].values
print(concepts)

# Isolating target into a separate array
target = data.iloc[:, -1].values
print(target)


def learn(concepts, target):
    '''
    learn() function implements the learning method of the Candidate elimination algorithm.
    Arguments:
        concepts - a numpy array with all the features
        target - a numpy array with corresponding output values
    '''

    # Initialize specific_h with the first instance from concepts
    specific_h = concepts[0].copy()
    print("\nInitialization of specific_h and general_h")
    print(specific_h)

    # Initialize general_h with '?' for all attributes
    general_h = [['?' for _ in range(len(specific_h))] for _ in range(len(specific_h))]
    print(general_h)

    # The learning iterations
    for i, h in enumerate(concepts):

        # Checking if the hypothesis has a positive target
        if target[i] == "Yes":
            for x in range(len(specific_h)):
                # Change values in S & G only if values change
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'

        # Checking if the hypothesis has a negative target
        if target[i] == "No":
            for x in range(len(specific_h)):
                # For negative hypothesis change values only in G
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'

        print("\nSteps of Candidate Elimination Algorithm", i + 1)
        print(specific_h)
        print(general_h)

    # find indices where we have empty rows, meaning those that are unchanged
    indices = [i for i, val in enumerate(general_h) if val == ['?' for _ in range(len(specific_h))]]
    for i in indices:
        # remove those rows from general_h
        general_h.remove(['?' for _ in range(len(specific_h))])
    # Return final values
    return specific_h, general_h


s_final, g_final = learn(concepts, target)
print("\nFinal Specific_h:", s_final, sep="\n")
print("\nFinal General_h:", g_final, sep="\n")
