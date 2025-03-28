# created by Reijer Klaasse
import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
resp = requests.get('https://www.thuisarts.nl/sitemap.xml?page=1')
root = ET.fromstring(resp.content)
urls = [url[0].text for url in root]
docs = []
def scrape_page(url):
    print(url)
    page = requests.get(url).content
    soup = BeautifulSoup(page, 'html.parser')
    page_title = soup.find('h1', class_='page-title').get_text(strip=True)
    blocks = soup.find_all('div', class_='field--name-field-ref-text-block')
    i = 0
    print(len(blocks))
    for block in blocks:
        h2 = block.find('h2')
        if h2 is not None:
            title = block.find('h2').get_text(strip=True)
            if title == "Film" or title == "Over deze tekst":
                continue
        else:
            continue
        text = block.find('article').find('div')
        if text is not None:
            text = text.get_text(strip=True)
        else:
            continue
        if len(text) < 100:
            continue
        file = open(f'./documents/{url.replace('https://www.thuisarts.nl/','').replace('/','')}-{i}.txt', 'w', encoding='utf-8')
    
        document = f'Document title: {page_title}\nURL: {url}\nParagraph title: {title}\nText: {text}\n\n'
        file.write(document)
        i += 1
        file.close()
for url in urls:
    scrape_page(url)