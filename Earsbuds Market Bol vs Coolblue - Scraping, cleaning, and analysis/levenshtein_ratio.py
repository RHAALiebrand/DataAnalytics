import numpy as np

# Let's set the strings
string1='I am string 1'
string2='I am string 2'

ratio_calc = True

# Initialize matrix of zeros
R = len(string1)+1
C = len(string2)+1
dist = np.zeros((R,C),dtype = int)

print(dist)

# Populate matrix of zeros with the indeces of each character of both strings
for i in range(1, R):
    for k in range(1,C):
        dist[i][0] = i
        dist[0][k] = k
print(dist)
# Sure name the above ugly

# Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions    
for col in range(1, C):
    for row in range(1, R):
        if string1[row-1] == string2[col-1]:
            cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
        else:
            if ratio_calc == True:
                cost = 2
            else:
                cost = 1
        dist[row][col] = min(dist[row-1][col] + 1,      # Cost of deletions
                             dist[row][col-1] + 1,          # Cost of insertions
                             dist[row-1][col-1] + cost)     # Cost of substitutions
    # Computation of the Levenshtein dist Ratio

Ratio = ((len(string1)+len(string2)) - dist[row][col]) / (len(string1)+len(string2))
print("The strings are {} edits away".format(dist[row][col]))
