import requests
import json
import schedule
import time
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

wx_3_api = 'https://data.tmd.go.th/api/Weather3Hours/V1/?type=json'

def getWX3(url):
    
    while True:
        try:
            response = requests.get(url,timeout=None)
            now = datetime.now()
            hour1 = int(now.strftime('%H'))-1
            date_time1 = now.strftime('%Y%m%d_{}00'.format(hour1))
            date_time2 = now.strftime('%a %d-%m-%Y %H:%M:%S')
            with open("{}_WX3.json".format(date_time1), "w", encoding='UTF-8') as outfile:
                json.dump(response.json(), outfile, indent=4, ensure_ascii=False)
            print('Collect WX3 data successfully - {}'.format(date_time2))
            
            break
    
        except:
            print('Collect data error!')
            time.sleep(20)

schedule.every().day.at('02:00').do(getWX3, wx_3_api)
schedule.every().day.at('05:00').do(getWX3, wx_3_api)
schedule.every().day.at('08:00').do(getWX3, wx_3_api)
schedule.every().day.at('11:00').do(getWX3, wx_3_api)
schedule.every().day.at('14:00').do(getWX3, wx_3_api)
schedule.every().day.at('17:00').do(getWX3, wx_3_api)
schedule.every().day.at('20:00').do(getWX3, wx_3_api)
schedule.every().day.at('23:00').do(getWX3, wx_3_api)
# schedule.every(10).seconds.do(getWX3, wx_3_api)

while True:
    schedule.run_pending()
    time.sleep(1)