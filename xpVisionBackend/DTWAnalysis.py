import numpy as np
import pandas as pd
import math

def FindDistance(num1, num2):
  return abs(num1 - num2)

#Overall complexity: O(signal1Len * signal2Len)
def DTW(signal1, signal2):

  # Get lengths of both input signals
  signal1Len = len(signal1)
  signal2Len = len(signal2)
  
  # Initialize cost matrix with infinity (no path computed yet)
  # costMatrix[i,j] = minimum cost to align signal1[0:i+1] with signal2[0:j+1]
  costMatrix = np.full((signal1Len, signal2Len), np.inf)
  
  # Initialize backtracking matrix to store the previous cell in optimal path
  # backTrackingMatrix[i,j] = (prev_i, prev_j) coordinates of previous step
  # -1 indicates no previous step (used for termination)
  backTrackingMatrix =  np.full((signal1Len, signal2Len, 2), -1)

  # Base case: cost of aligning first elements of both signals
  costMatrix[0, 0] = FindDistance(signal1[0], signal2[0])
  
  # Iterate through all possible alignments
  for i in range(signal1Len):
    for j in range(signal2Len):
      # Skip the base case (already computed)
      if(i == 0 and j == 0):
        continue

      # Build list of possible previous cells we could have come from
      # Each candidate is (cost_of_previous_cell, (previous_i, previous_j))
      candidates = []
      
      # Diagonal move: align signal1[i] with signal2[j], coming from [i-1,j-1]
      # This represents both signals advancing in time together
      if i > 0 and j > 0:
          candidates.append((costMatrix[i-1, j-1], (i-1, j-1)))
      
      # Vertical move: repeat signal2[j], coming from [i-1,j]
      # This represents signal1 advancing while signal2 stays at same position
      if i > 0:
          candidates.append((costMatrix[i-1, j], (i-1, j)))
      
      # Horizontal move: repeat signal1[i], coming from [i,j-1]
      # This represents signal2 advancing while signal1 stays at same position
      if j > 0:
          candidates.append((costMatrix[i, j-1], (i, j-1)))

      # Select the path with minimum cumulative cost (optimal substructure)
      min_cost, prev = min(candidates, key=lambda x: x[0])
      
      # Update cost matrix: current cell distance + minimum cost to reach here
      costMatrix[i, j] = FindDistance(signal1[i], signal2[j]) + min_cost
      
      # Store which cell we came from for backtracking
      backTrackingMatrix[i, j] = prev

  # Backtrack from bottom-right corner to find the optimal alignment path
  # Start at the end of both signals
  i, j = signal1Len - 1, signal2Len - 1
  path = []
  
  # Traverse backwards through the backtracking matrix
  while(i >= 0 and j >= 0):
      # Add current alignment point (signal1[i] aligns with signal2[j])
      path.append((int(i), int(j)))
      
      # Move to the previous cell in the optimal path
      i, j = backTrackingMatrix[i, j]
      
      # Stop when we reach the beginning (marked with -1)
      if(i == -1 or j == -1):
          break
  
  # Reverse path to get chronological order (start to end)
  path.reverse()
  
  # Return: final cost (bottom-right corner), optimal alignment path, full cost matrix
  return costMatrix[signal1Len - 1][signal2Len - 1], path, costMatrix