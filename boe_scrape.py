# Scraping BoE text data 
# %%

# Import Libraries
import requests
import pdfplumber
from bs4 import BeautifulSoup
from tqdm import tqdm
import os
import pandas as pd

# %%
# Get all speech links from sitemap
sitemap_url = "https://www.bankofengland.co.uk/sitemap/speeches"
headers = {"User-Agent": "Mozilla/5.0"}

r = requests.get(sitemap_url, headers=headers)
soup = BeautifulSoup(r.text, "html.parser")

# %%
# Extract all PDF URLs
pdf_links = [a["href"] for a in soup.select("a[href$='.pdf']")]

# %%
# Filter PDFs from 2015 onwards
pdf_links_filtered = [url for url in pdf_links if "/2015/" in url or
                      "/2016/" in url or "/2017/" in url or "/2018/" in url or
                      "/2019/" in url or "/2020/" in url or "/2021/" in url or
                      "/2022/" in url or "/2023/" in url or "/2024/" in url or
                      "/2025/" in url]

print(f"Found {len(pdf_links_filtered)} PDFs from 2015 onward.")

# %%
# Output directory
output_dir = "All_data/BoE_Speeches"

# Download and extract
for url in tqdm(pdf_links_filtered):
    try:
        filename = url.split("/")[-1]
        filepath = os.path.join(output_dir, filename)

        # Download PDF
        res = requests.get(url, headers=headers, stream=True)
        if "application/pdf" not in res.headers.get("Content-Type", ""):
            print(f"Skipping non-PDF: {url}")
            continue

        with open(filepath, "wb") as f:
            f.write(res.content)

        # Extract text
        with pdfplumber.open(filepath) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

        # Save as .txt file
        text_file = filepath.replace(".pdf", ".txt")
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(text)

    except Exception as e:
        print(f"Error processing {url}: {e}")


# %% 
# Extract metadata from link text in HTML
metadata = []

# Normalize filtered URLs (ensure full links)
pdf_links_filtered_set = set(
    url if url.startswith("http") else "https://www.bankofengland.co.uk" + url
    for url in pdf_links_filtered
)

for a in soup.select("a[href$='.pdf']"):
    href = a["href"]
    full_url = href if href.startswith("http") else "https://www.bankofengland.co.uk" + href

    if full_url in pdf_links_filtered_set:
        filename = full_url.split("/")[-1]

        # Extract anchor text before the <span> (if present)
        if a.find("span"):
            a.find("span").extract()  # Remove span to get clean title text

        link_text = a.get_text(strip=True)

        metadata.append({
            "filename": filename,
            "url": full_url,
            "link_text": link_text
        })

# Save to CSV
df = pd.DataFrame(metadata)
df.to_csv("All_data/Metadata/speech_metadata.csv", index=False)
print(f"Saved metadata for {len(df)} speeches.")



# %%

######## MINUTES 



# %% 
# Fetch the minutes sitemap
sitemap_url = "https://www.bankofengland.co.uk/sitemap/minutes"
headers = {"User-Agent": "Mozilla/5.0"}

res = requests.get(sitemap_url, headers=headers)
soup = BeautifulSoup(res.text, "html.parser")
# %%

# Get all <a> tags with .pdf links that include "monetary-policy-summary-and-minutes"
pdf_links = []
for a in soup.find_all("a", href=True):
    href = a["href"]
    if href.endswith(".pdf") and "monetary-policy-summary-and-minutes" in href.lower():
        pdf_links.append(href)

print(f"Found {len(pdf_links)} matching PDFs")

# %%
# Download and extract text from each PDF
output_dir = "All_data/BoE_Minutes"

for url in tqdm(pdf_links):
    try:
        filename = os.path.basename(url)
        filepath = os.path.join(output_dir, filename)
        r = requests.get(url, headers=headers, stream=True, allow_redirects=True)
        if "application/pdf" not in r.headers.get("Content-Type", ""):
            print(f"Skipping non‑PDF: {url}")
            continue
        with open(filepath, "wb") as f:
            f.write(r.content)
        # Optional: extract text
        with pdfplumber.open(filepath) as pdf:
            text = "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
        txt_path = filepath.replace(".pdf", ".txt")
        with open(txt_path, "w", encoding="utf8") as tf:
            tf.write(text)
    except Exception as e:
        print(f"Error on {url}: {e}")

# %%
# Get all <a> tags with links containing "monetary-policy-summary-and-minutes"
metadata = []

for a in soup.find_all("a", href=True):
    href = a["href"]
    if "monetary-policy-summary-and-minutes" in href.lower():
        full_url = href if href.startswith("http") else "https://www.bankofengland.co.uk" + href
        link_text = a.get_text(strip=True)

        metadata.append({
            "filename": os.path.basename(full_url),
            "url": full_url,
            "link_text": link_text,
        })

print(f"Found {len(metadata)} links with monetary-policy-summary-and-minutes")

# Save to CSV
df = pd.DataFrame(metadata)
df.to_csv("All_data/Metadata/minutes_metadata.csv", index=False)
print("Saved metadata to minutes_metadata.csv")
# %%
