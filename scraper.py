import requests
from bs4 import BeautifulSoup
import json
import pymongo
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import io
from urllib.parse import urljoin


# We'll use a helper to print safe text or reconfigure stdout if possible
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def get_listings_urls(region="cairo/new-cairo", page_number=1):
    """
    Fetches the list of property URLs from a specific region and page.
    """
    url = f"https://aqarmap.com.eg/en/for-sale/apartment/{region}/?page={page_number}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://aqarmap.com.eg/en/',
        'Connection': 'keep-alive'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return []

        soup = BeautifulSoup(response.content, 'html.parser')
        links = []
        
        # Extract listing links from the search results page
        base_url = "https://aqarmap.com.eg"
        for a in soup.find_all('a', href=True):
            if '/en/listing/' in a['href']:
                # urljoin handles both relative and absolute links correctly
                full_link = urljoin(base_url, a['href'])
                if full_link not in links:
                    links.append(full_link)
        
        return links
    except Exception as e:
        print(f"Error fetching page {page_number}: {e}")
        return []

def get_listing_details(listing_url):
    """
    Scrapes detailed information for a single property listing.
    Combines JSON-LD data and HTML parsing for maximum feature extraction.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        response = requests.get(listing_url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 1. Primary data extraction via hidden JSON-LD scripts
        script_tags = soup.find_all('script', type='application/ld+json')
        result = {'url': listing_url, 'source': 'aqarmap'}
        
        for script_tag in script_tags:
            if not script_tag.string:
                continue
                
            try:
                data = json.loads(script_tag.string)
                if isinstance(data, dict) and '@graph' in data and len(data['@graph']) > 0:
                    listing_info = data['@graph'][-1]
                    item_offered = listing_info.get('itemOffered') or {}
                    offers = listing_info.get('offers') or {}
                    geo = item_offered.get('geo') or {}
                    floor_size = item_offered.get('floorSize') or {}
                    
                    result.update({
                        'title': listing_info.get('name'),
                        'price': offers.get('price'),
                        'currency': offers.get('priceCurrency'),
                        'area': floor_size.get('value'),
                        'rooms': item_offered.get('numberOfRooms'),
                        'bathrooms': item_offered.get('numberOfBathroomsTotal'),
                        'latitude': geo.get('latitude'),
                        'longitude': geo.get('longitude')
                    })
                    break # Found the main listing JSON
            except Exception:
                continue

        # 2. Extract additional details from HTML (Floor, View, Payment Method, etc.)
        try:
            # Details section (Table-like key-value pairs)
            details_section = soup.find('section', id='details')
            if details_section:
                for div in details_section.find_all('div', class_='flex'):
                    key_tag = div.find('h4')
                    val_tag = div.find('span')
                    if key_tag and val_tag:
                        key = key_tag.get_text(strip=True).lower().replace(' ', '_')
                        val = val_tag.get_text(strip=True)
                        result[key] = val
            
            # Listing Description
            desc_section = soup.find('p', string=lambda text: text and 'Listing Description' in text)
            if desc_section:
                parent_section = desc_section.find_parent('section')
                if parent_section:
                    desc_div = parent_section.find('div', class_=lambda c: c and 'col-span-9' in c)
                    if desc_div:
                        result['description'] = desc_div.get_text(" ", strip=True)
            
            # Amenities (Checklist of features)
            amenities_section = soup.find('section', id='amenities')
            if amenities_section:
                amenities = []
                for li in amenities_section.find_all('li'):
                    span = li.find('span')
                    if span:
                        amenities.append(span.get_text(strip=True))
                result['amenities'] = amenities
        except Exception as e:
            print(f"Error parsing HTML details for {listing_url}: {e}")

        return result
    except Exception as e:
        print(f"Error scraping {listing_url}: {e}")
        return None

def main():
    # ==========================================
    # 1. CONFIGURATION & SETUP
    # ==========================================
    # ---- VALID AQARMAP REGIONS YOU CAN USE ----
    # Full Governorates: cairo, giza, alexandria, north-coast, red-sea, suez, qalyubia
    # Specific Cities (Example): cairo/new-cairo, giza/6-october, cairo/nasr-city
    # -------------------------------------------
    
    # CHANGE THIS VARIABLE TO SCRAPE A DIFFERENT REGION
    REGION = "north-coast"
    
    MAX_THREADS = 15  # Number of parallel workers to speed up scraping
    
    # ==========================================
    # 2. DISCOVERY PHASE: FIND ALL PROPERTY LINKS
    # ==========================================
    # Instead of just 2 pages, we will keep going until a page returns 0 properties.
    print(f"-> Starting search across all of Greater Cairo ({REGION})...")
    print("   (This might take a while because there are thousands of properties!)")
    
    all_links = []
    page = 1
    
    while True:
        links = get_listings_urls(REGION, page)
        
        # If we didn't find any links on this page, it means we reached the very last page!
        if not links:
            print(f"   Reached the end at page {page-1}. Stopping discovery.")
            break
            
        all_links.extend(links)
        print(f"   [Page {page}] Found {len(links)} links. (Total so far: {len(all_links)})")
        
        page += 1
        
        # Optional safeguard: Hard stop at 5000 pages so we don't loop forever by mistake.
        if page > 5000:
            print("   Reached safety limit of 5000 pages. Stopping.")
            break

    print(f"\n-> Discovery Complete! Total properties to scrape details for: {len(all_links)}")

    # ==========================================
    # 3. MONGODB CONNECTION (THE BIG DATA DATABASE)
    # ==========================================
    # Note on MongoDB: This is a NoSQL database running locally on your computer.
    # It allows us to store massive amounts of unstructured JSON without locking the schema.
    # If MongoDB isn't running on your laptop, the script safely falls back to just using a CSV file.
    client = None
    collection = None
    try:
        # Connecting to default local MongoDB port
        client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
        client.server_info()  # Check if connection is alive
        db = client["aqarmap_project"]
        collection = db["listings"]
        print("-> [SUCCESS] Connected to local MongoDB database. Data will be saved securely.")
    except Exception as e:
        print("-> [WARNING] Cannot connect to MongoDB. Is MongoDB Community Server running?")
        print(f"   Error: {e}")
        print("   Proceeding using CSV Export ONLY.")

    # ==========================================
    # 4. SCRAPING PHASE: FETCH DETAILS IN PARALLEL
    # ==========================================
    # Using ThreadPoolExecutor allows us to download multiple pages simultaneously.
    scraped_data = []
    print(f"-> Scraping details using {MAX_THREADS} threads...")
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        future_to_url = {executor.submit(get_listing_details, url): url for url in all_links}
        
        for i, future in enumerate(as_completed(future_to_url)):
            url = future_to_url[future]
            try:
                details = future.result()
                if details:
                    scraped_data.append(details)
                    
                    # Real-time save to MongoDB
                    # We use the URL as a unique ID so running the script twice won't duplicate data!
                    if collection is not None:
                        details['_id'] = details['url']
                        # upsert=True means: insert if it doesn't exist, update if it does.
                        collection.update_one({'_id': details['_id']}, {'$set': details}, upsert=True)
                    
                    # Print progress every 10 properties to avoid spamming the console
                    if (i + 1) % 10 == 0 or (i + 1) == len(all_links):
                        print(f"   Progress: {i+1}/{len(all_links)} properties scraped.")
            except Exception as e:
                print(f"   Error processing {url}: {e}")

    # ==========================================
    # 5. DATA EXPORT: SAVE TO CSV FOR CLEANING -> ML
    # ==========================================
    if scraped_data:
        # Convert list of dictionaries into a robust Pandas analytical table
        df = pd.DataFrame(scraped_data)
        
        # Flatten amenities lists (e.g. ['Elevator', 'Security']) into a single string
        if 'amenities' in df.columns:
            df['amenities'] = df['amenities'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
            
        # Clean region name for the file (e.g. "cairo/new-cairo" -> "cairo_new_cairo")
        safe_region = REGION.replace('/', '_').replace('-', '_')
        output_file = f"aqarmap_listings_{safe_region}.csv"
        
        # utf-8-sig ensures Excel safely reads any Arabic letters inside the CSV
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print("\n" + "="*50)
        print(f"[FINISHED] Completed scraping {len(scraped_data)} properties.")
        print(f"[DATA EXPORT] Saved to {output_file}")
        print("  -> Next step: Run cleaner.py to harmonize this data!")
        print("="*50)
    else:
        print("\n[FAILED] No data was scraped.")

if __name__ == "__main__":
    main()




