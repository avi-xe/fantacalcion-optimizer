import requests
from bs4 import BeautifulSoup

response = requests.get('https://www.pianetafanta.it/probabili-formazioni-complete-serie-a-live.asp')
with open("pianeta-fanta.html", "w+") as f:
    f.write(response.content.decode("utf-8"))

print(response.status_code)