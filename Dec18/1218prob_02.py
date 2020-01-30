
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics

df = pd.read_csv('HR_0.csv')

mean = statistics.mean(df['Weight in Kgs.'])

for i in range(len(df)):
    if df.at[i, 'Weight in Kgs.'] >= mean + 20.0:
        print('{} {}, {}kg (+{:.1f}kg)'.format(df.at[i, 'First Name'],
                                                        df.at[i, 'Last Name'],
                                                        df.at[i, 'Weight in Kgs.'],
                                                        df.at[i, 'Weight in Kgs.']-mean))
    
