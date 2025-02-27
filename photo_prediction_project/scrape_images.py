import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import shutil

# Create images directory
IMAGE_DIR = "images"
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# Define target websites
websites = {
    "time": "https://time.com/7176286/top-100-photos-2024/",
    "apnews": "https://apnews.com/associated-press-100-photos-of-2024-an-epic-catalog-of-humanity",
    "natgeo": "https://www.nationalgeographic.com/photography/graphics/pictures-of-the-year-2024",
    "worldpressphoto": "https://www.worldpressphoto.org/collection/photocontest/winners/2024",
    "nypost": "https://nypost.com/2024/12/31/the-best-photos-of-2024/",
    "atlantic": "https://www.theatlantic.com/photo/2024/12/top-25-news-photos-of-2024/1003081/",
    "reuters": "https://www.reuters.com/investigates/special-report/year-end-2024-photos-best/",
    'baw': 'https://bwphotoawards.com/en/2024-award-winning-photos/',
    "tdigpic": "https://community.the-digital-picture.com/showthread.php?t=9248"
}

# Function to download an image
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

# Scrape all websites
for site, link in websites.items():
    scrape_images(site, link)

print("Image scraping completed!")