import numpy as np
import matplotlib.pyplot as plt
index=[1,2,3]
health_average=[205.3268,20.44878,112.5233]
health_std=[22.14312,3.033226,17.12766]

ill_average=[179.2953,23.91529,102.2388]
ill_std=[40.35005,6.016838,25.61201]

plt.errorbar(index,health_average,yerr=health_std,fmt='*',ecolor='black',color='black',elinewidth=2,capsize=4)
plt.errorbar(index,ill_average,yerr=ill_std,fmt='o',ecolor='plum',markerfacecolor='none',color='plum',elinewidth=2,capsize=4)

plt.plot(index,health_average,color='green',label='Health')
plt.plot(index,ill_average,color='red',linestyle='--',label='HLB')

ax=plt.gca()
plt.xticks(index,['NIR','RED','GREEN'])
plt.ylabel('ROI Value')
plt.legend(loc='upper right')
plt.show()
