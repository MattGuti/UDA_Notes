url = 'https://apnews.com/associated-press-100-photos-of-2024-an-epic-catalog-of-humanity'

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
    
img_tags = soup.find_all('img')

img_tags[10].get('src')

download_image(img_tags[100].get('src'), "/Users/mattgutierrez80/Desktop/UDA_Notes/", "ap_test_100.jpg")
img_urls = [urljoin(url, img['src']) for img in img_tags if 'src' in img.attrs]

