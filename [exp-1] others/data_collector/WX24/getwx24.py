import requests
import json
import schedule
import time
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

wx_24_api = 'https://data.tmd.go.th/api/WeatherToday/V1/?type=json'

def getWX24(url):

    while True:
        try:
            response = requests.get(url)
            now = datetime.now()
            date1 = int(now.strftime('%d'))-1
            date_time1 = int(now.strftime('%Y%m{:02d}'.format(date1)))-1
            date_time2 = now.strftime('%a %d-%m-%Y %H:%M:%S')
            with open("{}_WX24.json".format(date_time1), "w", encoding='UTF-8') as outfile:
                json.dump(response.json(), outfile, indent=4, ensure_ascii=False)
            print('Collect WX24 data successfully - {}'.format(date_time2))

            break

        except:
            print('Collect data error!')
            time.sleep(20)

schedule.every().day.at('09:00').do(getWX24, wx_24_api)
# schedule.every(10).seconds.do(getWX24, wx_24_api)

while True:
    schedule.run_pending()
    time.sleep(1)