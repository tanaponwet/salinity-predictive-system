import requests
import json
import schedule
import time
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

rain_regions_api = 'https://data.tmd.go.th/api/RainRegions/v1/?uid=api&ukey=api12345&format=json'

def getRainRegions(url):
    response = requests.get(url)
    now = datetime.now()
    date1 = int(now.strftime('%d'))-1
    date_time1 = now.strftime('%Y%m{}'.format(date1))
    date_time2 = now.strftime('%a %d-%m-%Y %H:%M:%S')
    with open("{}_RR.json".format(date_time1), "w", encoding='UTF-8') as outfile:
        json.dump(response.json(), outfile, indent=4, ensure_ascii=False)
    print('collect RR data successfully - {}'.format(date_time2))

schedule.every().day.at('10:00').do(getRainRegions, rain_regions_api)
# schedule.every(10).seconds.do(getRainRegions, rain_regions_api)

while True:
    schedule.run_pending()
    time.sleep(1)