import os
import time
import shutil
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# ✅ Save images to this directory
IMAGE_DIR = "/Users/mattgutierrez80/Desktop/UDA_Notes/images/not_selected"
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# ✅ Set up Selenium WebDriver (Headless Mode)
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Run Chrome in the background
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920x1080")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# ✅ Global counter to track total images downloaded
downloaded_images = 0
MAX_IMAGES = 500

BASE_URL = "https://awkwardfamilyphotos.com/category/photos/random-awkwardness/page/{}/"

def download_image(img_url, folder, img_name):
    """Download an image from a given URL."""
    global downloaded_images
    if downloaded_images >= MAX_IMAGES:
        print("🚀 Reached 500 images! Stopping scraper.")
        return False  

    try:
        response = requests.get(img_url, stream=True, timeout=10)
        if response.status_code == 200:
            img_path = os.path.join(folder, img_name)
            with open(img_path, 'wb') as file:
                shutil.copyfileobj(response.raw, file)
            downloaded_images += 1
            print(f"✅ Downloaded ({downloaded_images}/{MAX_IMAGES}): {img_name}")
            return True
        else:
            print(f"❌ Failed to download: {img_url}")
    except Exception as e:
        print(f"❌ Error downloading {img_url}: {e}")
    return False

def scrape_pages():
    """Scrape images from Awkward Family Photos across multiple pages."""
    global downloaded_images
    page_number = 1  

    while downloaded_images < MAX_IMAGES:
        url = BASE_URL.format(page_number)
        print(f"🔍 Scraping Page {page_number}: {url}")

        driver.get(url)
        time.sleep(5)  

        img_elements = driver.find_elements(By.TAG_NAME, "img")
        img_urls = []

        for img in img_elements:
            src = img.get_attribute("data-src") or img.get_attribute("src")
            if src and "awkwardfamilyphotos" in src:  
                img_urls.append(src)

        if not img_urls:
            print(f"✅ No more images found on Page {page_number}. Stopping.")
            break  # ✅ Stop if no images are found

        print(f"🔍 Found {len(img_urls)} images on Page {page_number}.")

        for img_url in img_urls:
            if downloaded_images >= MAX_IMAGES:
                break
            download_image(img_url, IMAGE_DIR, f"awkward_{downloaded_images}.jpg")

        page_number += 1  # ✅ Go to the next page

# ✅ Run the scraper
scrape_pages()

# ✅ Close browser
driver.quit()

print("🎉 Image scraping completed! 500 images saved in:", IMAGE_DIR)

