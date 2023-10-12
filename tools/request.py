# coding: utf-8
import requests
import urllib.request
from urllib.parse import urlencode

api = "https://openai-api-yak3s7dv3a-ue.a.run.app/?q="
headers = {'User-Agent': 'Mozilla/5.0 (Windows;U;Windows NT 6.1;en-US;rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6'}
params = {
    "q": "以助手的身份回复'你真棒'"
}

if __name__ == "__main__":
    res = requests.get(api)

    headers = {'User-Agent': 'Mozilla/5.0 (Windows;U;Windows NT 6.1;en-US;rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6'}
    request = urllib.request.urlopen(url=api).read()
    print(request)
    print(res.url)
    print(res.text)