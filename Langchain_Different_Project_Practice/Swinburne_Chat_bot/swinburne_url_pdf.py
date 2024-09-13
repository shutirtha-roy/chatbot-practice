import requests
from bs4 import BeautifulSoup
import os

def fetch_html(url):
    """Fetch HTML content from a given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def extract_body_content(html_content):
    """Extract body content from HTML."""
    soup = BeautifulSoup(html_content, 'html.parser')
    body = soup.find('body')
    if body:
        # Remove script and style elements
        for script in body(["script", "style"]):
            script.decompose()
        return body.get_text(strip=True)
    return ""

def save_content_as_text(content, filename):
    """Save content as a text file."""
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"Content saved successfully: {filename}")
    except Exception as e:
        print(f"Error saving content: {e}")

def process_url(url, output_dir):
    """Process a single URL: fetch, extract, and save content."""
    html_content = fetch_html(url)
    if html_content:
        body_content = extract_body_content(html_content)
        safe_filename = "".join([c for c in url if c.isalpha() or c.isdigit() or c==' ']).rstrip()
        output_file = os.path.join(output_dir, f"{safe_filename}.txt")
        save_content_as_text(body_content, output_file)

def main():
    urls = [
        'https://www.swinburne.edu.au/',
        'https://www.swinburne.edu.au/about/our-university/rankings-ratings/',
        'https://www.swinburne.edu.au/about/policies-regulations/',
    ]
    
    output_dir = "extracted_content"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, url in enumerate(urls, 1):
        print(f"Processing file: {i}")
        process_url(url, output_dir)
        print(f"File {i} processed")

if __name__ == "__main__":
    main()