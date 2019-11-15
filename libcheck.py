#check model.selection
try:
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import KFold
    print("Sucessed in importing model selection")
except:
    print("Failed to import model selection")

#check lib matplotlib
try:
    import matplotlib.pyplot as plt 
    print("Sucessed in importing matplotlib")
except:
    print("Failed to import matplotlib")

#check lib NumPy
try:
    import numpy as np
    print("Sucessed in importing NumPy")
except:
    print("Failed to import NumPy")
#check lib Pandas
try:
    import pandas as pd
    print("Sucessed in importing Pandas")
except:
    print("Failed to import Pandas")