import numpy as np
import pandas as pd
import math

def FindDistance(num1, num2):
  return abs(num1 - num2)

def DTW(signal1, signal2):
  signal1Len = len(signal1)
  signal2Len = len(signal2)
  costMatrix = np.full((signal1Len, signal2Len), np.inf)
  backTrackingMatrix =  np.full((signal1Len, signal2Len, 2), -1)

  costMatrix[0, 0] = FindDistance(signal1[0], signal2[0])
  for i in range(signal1Len):
    for j in range(signal2Len):
      if(i == 0 and j == 0):
        continue

      candidates = []
      if i > 0 and j > 0:
          candidates.append((costMatrix[i-1, j-1], (i-1, j-1)))
      if i > 0:
          candidates.append((costMatrix[i-1, j], (i-1, j)))
      if j > 0:
          candidates.append((costMatrix[i, j-1], (i, j-1)))

      min_cost, prev = min(candidates, key=lambda x: x[0])
      costMatrix[i, j] = FindDistance(signal1[i], signal2[j]) + min_cost
      backTrackingMatrix[i, j] = prev

  
  i, j = signal1Len - 1, signal2Len - 1
  path = []
  while(i >= 0 and j >= 0):
      path.append((int(i), int(j)))
      i, j = backTrackingMatrix[i, j]
      if(i == -1 or j == -1):
          break
      path.append((int(i), int(j)))
  path.reverse()
  print(path)
  return costMatrix[signal1Len - 1][signal2Len - 1]