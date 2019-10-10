# import requests
# import json
# import prettytable
# headers = {'Content-type': 'application/json'}
# data = json.dumps({"seriesid": ['LNS14000000'],"startyear":"2019", "endyear":"2019"})
# p = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data, headers=headers)
# json_data = json.loads(p.text)
# for series in json_data['Results']['series']:
#     x=prettytable.PrettyTable(["series id","year","period","value","footnotes"])
#     seriesId = series['seriesID']
#     for item in series['data']:
#         year = item['year']
#         period = item['period']
#         value = item['value']
#         footnotes=""
#         for footnote in item['footnotes']:
#             if footnote:
#                 footnotes = footnotes + footnote['text'] + ','
#         if 'M01' <= period <= 'M12':
#             x.add_row([seriesId,year,period,value,footnotes[0:-1]])
#     output = open(seriesId +'2019_2019' '.txt','w')
#     output.write (x.get_string())
#     output.close()

# 10년씩밖에 안읽어와짐.

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd

series = pd.read_csv('unemployment rate_2010_2019.csv', header=0, index_col=0, squeeze=True)
print(series)
diff_1=series.diff(periods=1).iloc[1:]
# diff_1.plot()
# plot_acf(diff_1)
# plot_pacf(diff_1)
# plt.show()
# ---------------
# series.plot()
# plot_acf(series)
# plot_pacf(series)
# plt.show()

from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(series, order=(0,1,1))
model_fit = model.fit(trend='nc',full_output=True, disp=1)
print(model_fit.summary())

model_fit.plot_predict()
plt.show()

fore = model_fit.forecast(steps=1)
print(fore)