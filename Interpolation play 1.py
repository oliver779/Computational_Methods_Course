# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 15:48:14 2021

@author: hlack
"""

import pandas as pd
import numpy as np
lst1 = [2,np.nan,5,8,4,9]
lst2 = [1,3,5,np.nan,np.nan,2]
lst3 = [1,np.nan,4,np.nan,2,1]

df = pd.DataFrame(list(zip(lst1,lst2,lst3)))


f = df.interpolate()
#df.interpolate(method='linear', limit_direction='forward')
print(df)
print(f)