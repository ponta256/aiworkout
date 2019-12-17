
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics
import random

df = pd.read_csv('HR_0.csv')

for i in range(len(df)):
    df.at[i,'Age in Yrs.'] += random.randint(0, 10)

df.to_csv('out.csv', index=False)
