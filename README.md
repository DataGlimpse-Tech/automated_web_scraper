# 🔸 Automated Web Scraper

This project is a fully automated, customizable web scraping tool that streamlines the entire process — from querying search engines to scraping structured data across paginated results. Designed to minimize manual intervention, it offers a seamless interface to extract large-scale data efficiently.

## 🚀 Features

* 🔍 **Automated SERP Search** using keywords
* 🌐 **URL Extraction** from search engine results
* ✨ **Smart Site Selection** based on data quantity
* 🧠 **Field Suggestion Engine** for dynamic scraping setup
* 📄 **Pagination Handling** to scrape all pages automatically
* 📦 **Data Export** in CSV format
* 🧰 **Modular Design** for easy customization

## 📸 App Workflow

1. **Input Search Query**
   Provide your target keywords (e.g., "Best laptops 2025").

2. **SERP API Integration**
   Automatically fetch URLs from Google/Bing using the SERP API.

3. **Site Selection**
   The system highlights sites with the most data-rich structures.

4. **Field Suggestion**
   Get suggested fields (e.g., title, price, ratings) using ML-based tagging or DOM inspection.

5. **Field Selection & Confirmation**
   Choose the fields you want to scrape.

6. **Scraping & Pagination**
   Data is scraped from all paginated results without user input.

7. **Download Results**
   Get structured data in CSV/JSON format.

## 🛠️ Tech Stack

* **Frontend**: Streamlit
* **Backend**: Python
* **APIs**: SERP API, Gemini API
* **Libraries**:

  * `BeautifulSoup` / `lxml` for parsing
  * `requests` for HTTP
  * `pandas` for data handling
  * `re` for regex-based field matching
  * `selenium` for dynamic web scraping and JavaScript-rendered pages
  * `json`, `csv` for export

## 📦 Installation

```bash
git clone https://github.com/DataGlimpse-Tech/automated_web_scraper.git
cd automated_web_scraper
pip install -r requirements.txt
streamlit run app.py
```
