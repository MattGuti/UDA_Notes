import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import shutil
import numpy as np

IMAGE_DIR = "/Users/mattgutierrez80/Desktop/UDA_Notes/images/not_selected"
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# Target Websites
websites = {
    "pexels": "https://www.pexels.com/search/random/"
    }

def download_image(img_url, folder, img_name):
    response = requests.get(img_url, stream=True)
    if response.status_code == 200:
        img_path = os.path.join(folder, img_name)
        with open(img_path, 'wb') as file:
            shutil.copyfileobj(response.raw, file)
        print(f"Downloaded: {img_name}")
    else:
        print(f"Failed to download: {img_url}")

# Function to scrape images from a website
def scrape_images(site_name, url):
    print(f"Scraping images from {site_name}...")
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    img_tags = soup.find_all('img')
    img_urls = [urljoin(url, img['src']) for img in img_tags if 'src' in img.attrs]

    for i, img_url in enumerate(img_urls):
        download_image(img_url, IMAGE_DIR, f"{site_name}_{i}.jpg")

# Scraping websites
for site, link in websites.items():
    np.random.uniform(.1, .25, 1)
    scrape_images(site, link)

print("Image scraping completed!")