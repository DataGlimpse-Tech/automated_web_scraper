# streamlit_app.py

import streamlit as st
from streamlit_tags import st_tags_sidebar
import pandas as pd
import json
from datetime import datetime
from scraper import (
    fetch_html_selenium,
    save_raw_data,
    format_data,
    save_formatted_data,
    html_to_markdown_with_readability,
    create_dynamic_listing_model,
    create_listings_container_model,
    setup_selenium,
    generate_unique_folder_name
)
from pagination_detector import detect_pagination_elements
from automated_scraper import run_automated_pagination_scrape
import re
from urllib.parse import urlparse

import os
import google.generativeai as genai
import requests


st.set_page_config(page_title="Web Scraper with Search")
st.title("Web Scraper with Search")


if 'scraping_state' not in st.session_state:
    st.session_state['scraping_state'] = 'idle'  
if 'results' not in st.session_state:
    st.session_state['results'] = None
if 'driver' not in st.session_state:
    st.session_state['driver'] = None
if 'suggested_fields' not in st.session_state:
    st.session_state['suggested_fields'] = []
if 'field_suggestions_loading' not in st.session_state:
    st.session_state['field_suggestions_loading'] = False
if 'extract_all_fields' not in st.session_state:
    st.session_state['extract_all_fields'] = False
if 'search_results' not in st.session_state:
    st.session_state['search_results'] = None
if 'search_mode' not in st.session_state:
    st.session_state['search_mode'] = 'manual'
if 'searching' not in st.session_state:
    st.session_state['searching'] = False


def search_urls(query, serp_api_key, num_results=10):
    """Search for URLs using SERP API based on user query"""
    if not serp_api_key or not query:
        return []
    
    try:

        url = "https://serpapi.com/search"
        
        params = {
            "engine": "google",
            "q": query,
            "api_key": serp_api_key,
            "num": num_results,
            "hl": "en",
            "gl": "us"
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        

        search_results = []
        if "organic_results" in data:
            for result in data["organic_results"]:
                search_results.append({
                    "title": result.get("title", ""),
                    "url": result.get("link", ""),
                    "snippet": result.get("snippet", ""),
                    "position": result.get("position", 0)
                })
        
        return search_results
    
    except requests.exceptions.RequestException as e:
        st.error(f"Error making request to SERP API: {str(e)}")
        return []
    except Exception as e:
        st.error(f"Error searching for URLs: {str(e)}")
        return []


def get_field_suggestions(url, gemini_api_key):
    """Analyze website content and suggest extractable fields"""
    if not gemini_api_key:
        return []
    
    try:

        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
 
        raw_html = fetch_html_selenium(url, attended_mode=False)
        markdown_content = html_to_markdown_with_readability(raw_html)
        
  
        if len(markdown_content) > 15000:
            markdown_content = markdown_content[:15000] + "..."
        
        prompt = f"""
        Analyze the following website content and suggest relevant fields that could be extracted for data scraping.
        Focus on identifying structured data elements like product information, listings, articles, prices, etc.
        
        Website content:
        {markdown_content}
        
        Please provide a JSON array of suggested field names that would be most useful to extract from this website.
        Consider common data patterns like:
        - Product names, prices, descriptions
        - Article titles, authors, dates
        - Contact information
        - Location data
        - Reviews and ratings
        - Images and links
        - Categories and tags
        
        Return only a JSON array of 8-15 relevant field names, nothing else.
        Example format: ["title", "price", "description", "image_url", "rating"]
        """
        
        response = model.generate_content(prompt)
        
 
        import json
        try:
            suggestions = json.loads(response.text.strip())
            if isinstance(suggestions, list):
                return suggestions[:15]  
        except json.JSONDecodeError:

            import re
            field_pattern = r'"([^"]+)"'
            matches = re.findall(field_pattern, response.text)
            return matches[:15] if matches else []
            
    except Exception as e:
        st.error(f"Error getting field suggestions: {str(e)}")
        return []
    
    return []


def extract_all_fields_from_content(url, gemini_api_key):
    """Analyze website content and extract all available structured data fields"""
    if not gemini_api_key:
        return []
    
    try:
 
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
   
        raw_html = fetch_html_selenium(url, attended_mode=False)
        markdown_content = html_to_markdown_with_readability(raw_html)
        
  
        if len(markdown_content) > 20000:
            markdown_content = markdown_content[:20000] + "..."
        
        prompt = f"""
        Analyze the following website content and identify ALL possible structured data fields that can be extracted.
        Look for every piece of structured information including but not limited to:
        
        - Product/item information (names, prices, descriptions, SKUs, brands, etc.)
        - Article/content data (titles, authors, dates, categories, tags, etc.)
        - Contact and location information
        - Media elements (images, videos, links)
        - Ratings, reviews, and feedback
        - Specifications and attributes
        - Availability and inventory data
        - Social media links and shares
        - Navigation and menu items
        - Any other structured data elements
        
        Website content:
        {markdown_content}
        
        Please provide a comprehensive JSON array of ALL extractable field names from this website.
        Be thorough and include every possible data field you can identify.
        Return only a JSON array of field names, nothing else.
        Example format: ["title", "price", "description", "image_url", "rating", "brand", "category", "availability", "author", "date", "tags"]
        """
        
        response = model.generate_content(prompt)
        

        try:
            all_fields = json.loads(response.text.strip())
            if isinstance(all_fields, list):
                return all_fields[:50]  
        except json.JSONDecodeError:
            
            field_pattern = r'"([^"]+)"'
            matches = re.findall(field_pattern, response.text)
            return matches[:50] if matches else []
            
    except Exception as e:
        st.error(f"Error extracting all fields: {str(e)}")
        return []
    
    return []


st.sidebar.title("Web Scraper Settings")


with st.sidebar.expander("API Keys", expanded=False):
    gemini_api_key = st.text_input("Gemini API Key", type="password")
    if gemini_api_key:
        st.session_state['gemini_api_key'] = gemini_api_key
    
    serp_api_key = st.text_input("SERP API Key", type="password", help="Get your API key from https://serpapi.com/")
    if serp_api_key:
        st.session_state['serp_api_key'] = serp_api_key

# Set model to Gemini 1.5 (no selection option)
model_selection = "gemini-1.5-flash"  # Fixed to Gemini 1.5

# Search Mode Selection
st.sidebar.markdown("### Input Mode")
search_mode = st.sidebar.radio(
    "Choose input method:",
    ["Manual URL Entry", "Search for URLs"],
    key="search_mode_radio"
)

# Update search mode in session state
if search_mode == "Search for URLs":
    st.session_state['search_mode'] = 'search'
else:
    st.session_state['search_mode'] = 'manual'

# Input based on mode
if st.session_state['search_mode'] == 'search':
    # Search mode
    st.sidebar.markdown("### Search for URLs")
    search_query = st.sidebar.text_input(
        "Enter search query for dataset:",
        placeholder="e.g., 'best restaurants in New York', 'iPhone 15 reviews', 'real estate listings'"
    )
    
    num_search_results = st.sidebar.slider(
        "Number of search results",
        min_value=1,
        max_value=20,
        value=10,
        help="Number of URLs to search for"
    )
    
    # Search button
    if st.sidebar.button("🔍 Search for URLs", type="primary"):
        if not search_query.strip():
            st.error("Please enter a search query.")
        elif not st.session_state.get('serp_api_key'):
            st.error("Please enter your SERP API key to search for URLs.")
        else:
            st.session_state['searching'] = True
            st.rerun()
    
    # Perform search
    if st.session_state.get('searching', False):
        with st.sidebar:
            with st.spinner("Searching for URLs..."):
                search_results = search_urls(
                    search_query, 
                    st.session_state['serp_api_key'], 
                    num_search_results
                )
                st.session_state['search_results'] = search_results
                st.session_state['searching'] = False
                
                if search_results:
                    st.success(f"Found {len(search_results)} URLs!")
                else:
                    st.error("No URLs found. Try a different search query.")
                st.rerun()
    
    # Display search results
    if st.session_state.get('search_results'):
        st.sidebar.markdown("### Search Results")
        search_results = st.session_state['search_results']
        
        # Show first few results with selection
        selected_urls = []
        for i, result in enumerate(search_results[:5]):  # Show first 5 results
            with st.sidebar.expander(f"Result {i+1}: {result['title'][:50]}..."):
                st.write(f"**URL:** {result['url']}")
                st.write(f"**Snippet:** {result['snippet'][:100]}...")
                if st.checkbox(f"Select for scraping", key=f"select_url_{i}"):
                    selected_urls.append(result['url'])
        
        # Auto-select first URL if none selected
        if not selected_urls and search_results:
            url_input = search_results[0]['url']
            st.sidebar.info(f"Auto-selected first result: {search_results[0]['title'][:50]}...")
        else:
            url_input = ' '.join(selected_urls)
    else:
        url_input = ""
        
else:
    # Manual mode
    st.sidebar.markdown("### Manual URL Entry")
    url_input = st.sidebar.text_input("Enter URL(s) separated by whitespace")

# Extract All Fields Option
extract_all_fields = st.sidebar.toggle("🎯 Extract All Fields", help="Automatically detect and extract all available structured data fields from the website")

# Field Suggestions Button (only show if not extracting all fields and have URL)
if (not extract_all_fields and 
    ((st.session_state['search_mode'] == 'manual' and url_input.strip()) or 
     (st.session_state['search_mode'] == 'search' and st.session_state.get('search_results'))) and 
    st.session_state.get('gemini_api_key')):
    
    # Get the URL for field suggestions
    if st.session_state['search_mode'] == 'search' and st.session_state.get('search_results'):
        suggestion_url = st.session_state['search_results'][0]['url']
    else:
        urls = url_input.strip().split()
        suggestion_url = urls[0] if urls else None
    
    if suggestion_url:
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            if st.button("🔍 Get Field Suggestions", help="Analyze the website and suggest extractable fields"):
                st.session_state['field_suggestions_loading'] = True
                st.rerun()
        
        # Show loading state
        if st.session_state['field_suggestions_loading']:
            with st.sidebar:
                with st.spinner("Analyzing website content..."):
                    suggestions = get_field_suggestions(suggestion_url, st.session_state['gemini_api_key'])
                    st.session_state['suggested_fields'] = suggestions
                    st.session_state['field_suggestions_loading'] = False
                    if suggestions:
                        st.success(f"Found {len(suggestions)} field suggestions!")
                    else:
                        st.warning("No field suggestions found.")
                    st.rerun()

# Debug info in sidebar
if st.session_state['search_mode'] == 'manual' and not extract_all_fields:
    st.sidebar.markdown("### 🔍 Field Suggestions Debug")
    if not url_input.strip():
        st.sidebar.info("Enter a URL to enable field suggestions")
    elif not st.session_state.get('gemini_api_key'):
        st.sidebar.warning("Enter Gemini API key to enable field suggestions")
    else:
        st.sidebar.success("Field suggestions available! Click 'Get Field Suggestions' button above.")

# Display suggested fields if available and not extracting all fields
if not extract_all_fields and st.session_state['suggested_fields']:
    st.sidebar.markdown("### 💡 Suggested Fields")
    st.sidebar.markdown("*Click on fields to add them to extraction:*")
    
    # Create clickable buttons for each suggestion
    cols = st.sidebar.columns(2)
    for i, field in enumerate(st.session_state['suggested_fields']):
        col = cols[i % 2]
        with col:
            if st.button(f"➕ {field}", key=f"suggest_{i}", help=f"Add '{field}' to extraction fields"):
                # Add to session state for fields input
                if 'fields_input' not in st.session_state:
                    st.session_state['fields_input'] = []
                if field not in st.session_state['fields_input']:
                    st.session_state['fields_input'].append(field)
                    st.rerun()

# Process URLs
if st.session_state['search_mode'] == 'search' and st.session_state.get('search_results'):
    # Use selected URLs from search results
    urls = url_input.strip().split() if url_input.strip() else [st.session_state['search_results'][0]['url']]
else:
    urls = url_input.strip().split() if url_input.strip() else []

num_urls = len(urls)

# Fields to extract
show_tags = st.sidebar.toggle("Enable Scraping")
fields = []

if show_tags:
    if extract_all_fields:
        # Show info about extract all fields mode
        st.sidebar.info("🎯 **Extract All Fields Mode**: The scraper will automatically detect and extract all available structured data fields from the website.")
        fields = ["auto_extract_all"]  # Special marker for all fields extraction
    else:
        # Use suggested fields as suggestions
        current_suggestions = st.session_state.get('suggested_fields', [])
        
        fields = st_tags_sidebar(
            label='Enter Fields to Extract:',
            text='Press enter to add a field',
            value=st.session_state.get('fields_input', []),
            suggestions=current_suggestions,
            maxtags=-1,
            key='fields_input'
        )

st.sidebar.markdown("---")

# Conditionally display Pagination and Attended Mode options
if num_urls <= 1:
    # Pagination settings
    use_pagination = st.sidebar.toggle("Enable Pagination")
    pagination_details = ""
    auto_paginate = False
    max_pages = 20
    if use_pagination:
        pagination_details = st.sidebar.text_input(
            "Enter Pagination Details (optional)",
            help="Describe how to navigate through pages (e.g., 'Next' button class, URL pattern)"
        )
        
        # Add automated pagination option
        auto_paginate = st.sidebar.toggle(
            "🤖 Automated Pagination", 
            value=True,
            help="Automatically discover and scrape all paginated pages without manual intervention"
        )
        
        if auto_paginate:
            max_pages = st.sidebar.slider(
                "Max Pages to Scrape", 
                min_value=1, 
                max_value=100, 
                value=5,
                help="Limit the number of pages to scrape (recommended: start with 2-5 pages for testing)"
            )
            
            # Add page range selection
            col1, col2 = st.sidebar.columns(2)
            with col1:
                start_page = st.number_input(
                    "Start Page", 
                    min_value=1, 
                    value=1,
                    help="Which page to start scraping from"
                )
            with col2:
                end_page = st.number_input(
                    "End Page", 
                    min_value=start_page, 
                    max_value=max_pages + start_page - 1,
                    value=min(max_pages, 5),
                    help="Which page to stop scraping at"
                )
                
            actual_max_pages = end_page - start_page + 1
            st.sidebar.info(f"🚀 Will scrape pages {start_page} to {end_page} ({actual_max_pages} pages total)")
            
            # Page count info
            st.sidebar.info(f"📄 Will process {actual_max_pages} pages total")
        else:
            max_pages = 20
            start_page = 1
            end_page = max_pages
            st.sidebar.info("Manual mode will only detect pagination URLs for you to review")

    st.sidebar.markdown("---")

    # Attended mode toggle
    attended_mode = st.sidebar.toggle("Enable Attended Mode")
else:
    # Multiple URLs entered; disable Pagination and Attended Mode
    use_pagination = False
    attended_mode = False
    pagination_details = ""  
    auto_paginate = False
    max_pages = 20
    start_page = 1
    end_page = 5
    actual_max_pages = 5
    st.sidebar.info("Pagination and Attended Mode are disabled when multiple URLs are entered.")
    if st.session_state['search_mode'] == 'manual':
        st.sidebar.info("Field suggestions are only available for single URLs.")

st.sidebar.markdown("---")

# Main action button
scrape_button_text = "LAUNCH SCRAPER"
if st.session_state['search_mode'] == 'search':
    scrape_button_text = "SCRAPE SELECTED URLS"

if st.sidebar.button(scrape_button_text, type="primary"):
    if not urls:
        if st.session_state['search_mode'] == 'search':
            st.error("Please search for URLs first.")
        else:
            st.error("Please enter at least one URL.")
    elif show_tags and not extract_all_fields and len(fields) == 0:
        st.error("Please enter at least one field to extract or enable 'Extract All Fields'.")
    elif show_tags and not st.session_state.get('gemini_api_key'):
        st.error("Please enter your Gemini API key to use scraping features.")
    else:
        # Set up scraping parameters in session state
        st.session_state['urls'] = urls
        st.session_state['fields'] = fields
        st.session_state['extract_all_fields'] = extract_all_fields
        st.session_state['model_selection'] = model_selection  # Will always be "gemini-1.5-flash"
        st.session_state['attended_mode'] = attended_mode
        st.session_state['use_pagination'] = use_pagination
        st.session_state['pagination_details'] = pagination_details
        st.session_state['auto_paginate'] = auto_paginate
        st.session_state['max_pages'] = actual_max_pages if auto_paginate else max_pages
        st.session_state['start_page'] = start_page if auto_paginate else 1
        st.session_state['end_page'] = end_page if auto_paginate else max_pages
        st.session_state['scraping_state'] = 'waiting' if attended_mode else 'scraping'

# Main content area - Show search results if in search mode
if st.session_state['search_mode'] == 'search' and st.session_state.get('search_results'):
    st.subheader("🔍 Search Results")
    
    search_results = st.session_state['search_results']
    
    # Create a DataFrame for better display
    display_data = []
    for i, result in enumerate(search_results):
        display_data.append({
            "Position": i + 1,
            "Title": result['title'],
            "URL": result['url'],
            "Snippet": result['snippet'][:150] + "..." if len(result['snippet']) > 150 else result['snippet']
        })
    
    df = pd.DataFrame(display_data)
    
    st.dataframe(
        df,
        column_config={
            "URL": st.column_config.LinkColumn("URL"),
            "Position": st.column_config.NumberColumn("Position", width="small"),
            "Title": st.column_config.TextColumn("Title", width="large"),
            "Snippet": st.column_config.TextColumn("Snippet", width="large")
        },
        use_container_width=True,
        hide_index=True
    )
    
    # Show which URL will be scraped
    if urls:
        st.info(f"🎯 Ready to scrape: **{urls[0]}**")
        st.write(f"**Title:** {search_results[0]['title']}")
        st.write(f"**Snippet:** {search_results[0]['snippet']}")
    
    # Download search results
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download Search Results (CSV)",
            data=df.to_csv(index=False),
            file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    with col2:
        st.download_button(
            "Download Search Results (JSON)",
            data=json.dumps(search_results, indent=2),
            file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# Scraping logic (rest of the code remains the same)
if st.session_state['scraping_state'] == 'waiting':
    # Attended mode: set up driver and wait for user interaction
    if st.session_state['driver'] is None:
        st.session_state['driver'] = setup_selenium(attended_mode=True)
        st.session_state['driver'].get(st.session_state['urls'][0])
        st.write("Perform any required actions in the browser window that opened.")
        st.write("Navigate to the page you want to scrape.")
        st.write("When ready, click the 'Resume Scraping' button.")
    else:
        st.write("Browser window is already open. Perform your actions and click 'Resume Scraping'.")

    if st.button("Resume Scraping"):
        st.session_state['scraping_state'] = 'scraping'
        st.rerun()

elif st.session_state['scraping_state'] == 'scraping':
    with st.spinner('Scraping in progress...'):
        # Perform scraping
        output_folder = os.path.join('output', generate_unique_folder_name(st.session_state['urls'][0]))
        os.makedirs(output_folder, exist_ok=True)

        all_data = []
        pagination_info = None

        driver = st.session_state.get('driver', None)
        if st.session_state['attended_mode'] and driver is not None:
            # Attended mode: scrape the current page without navigating
            # Fetch HTML from the current page
            raw_html = fetch_html_selenium(st.session_state['urls'][0], attended_mode=True, driver=driver)
            markdown = html_to_markdown_with_readability(raw_html)
            save_raw_data(markdown, output_folder, f'rawData_1.md')

            current_url = driver.current_url  # Use the current URL for logging and saving purposes

            # Detect pagination if enabled
            if st.session_state['use_pagination']:
                try:
                    pagination_data, token_counts = detect_pagination_elements(
                        current_url, st.session_state['pagination_details'], st.session_state['model_selection'], markdown
                    )
                    # Check if pagination_data is a dict or a model with 'page_urls' attribute
                    if isinstance(pagination_data, dict):
                        page_urls = pagination_data.get("page_urls", [])
                    else:
                        page_urls = pagination_data.page_urls if pagination_data else []
                    
                    pagination_info = {
                        "page_urls": page_urls,
                        "token_counts": token_counts
                    }
                except Exception as e:
                    st.warning(f"Pagination detection failed: {str(e)}")
                    pagination_info = None
                    
            # Scrape data if fields are specified
            if show_tags:
                # Handle extract all fields mode
                if st.session_state['extract_all_fields']:
                    # Get all available fields from the content
                    all_fields = extract_all_fields_from_content(current_url, st.session_state['gemini_api_key'])
                    if all_fields:
                        actual_fields = all_fields
                        st.info(f"🎯 Extracted {len(all_fields)} fields: {', '.join(all_fields[:10])}{'...' if len(all_fields) > 10 else ''}")
                    else:
                        # Fallback to common fields if auto-detection fails
                        actual_fields = ["title", "description", "price", "image_url", "link", "category", "rating", "availability"]
                        st.warning("Auto field detection failed. Using common fallback fields.")
                else:
                    actual_fields = st.session_state['fields']
                
                # Create dynamic models
                DynamicListingModel = create_dynamic_listing_model(actual_fields)
                DynamicListingsContainer = create_listings_container_model(DynamicListingModel)
                
                # Format data with error handling
                try:
                    formatted_data, token_counts = format_data(
                        markdown, DynamicListingsContainer, DynamicListingModel, st.session_state['model_selection']
                    )
                    
                    # Save formatted data with error handling
                    df = save_formatted_data(formatted_data, output_folder, f'sorted_data_1.json', f'sorted_data_1.xlsx')
                    
                    if df is not None:
                        all_data.append(formatted_data)
                        st.success("✅ Data extracted and saved successfully!")
                    else:
                        st.warning("⚠️ Data was saved but DataFrame creation failed")
                        all_data.append(formatted_data)
                        
                except Exception as e:
                    st.error(f"❌ Error during data extraction: {str(e)}")
                    st.info("💡 This might be due to:")
                    st.write("- Invalid response from AI model")
                    st.write("- Network connectivity issues")  
                    st.write("- Malformed website content")
                    st.write("- API rate limits")
                    # Continue with empty data to avoid complete failure
                    all_data.append({"error": str(e), "listings": []})
        else:
            # Non-attended mode or driver not available
            
            # Check if we should use automated pagination for single URL
            if (st.session_state['use_pagination'] and 
                st.session_state.get('auto_paginate', False) and 
                len(st.session_state['urls']) == 1 and 
                show_tags):
                
                # Use automated pagination scraper
                st.info("🤖 Running automated pagination scraping...")
                
                # Determine fields to extract
                if st.session_state['extract_all_fields']:
                    # Get all available fields from the content
                    all_fields = extract_all_fields_from_content(st.session_state['urls'][0], st.session_state['gemini_api_key'])
                    if all_fields:
                        actual_fields = all_fields
                        st.info(f"🎯 Auto-detected {len(all_fields)} fields: {', '.join(all_fields[:10])}{'...' if len(all_fields) > 10 else ''}")
                    else:
                        # Fallback to common fields if auto-detection fails
                        actual_fields = ["title", "description", "price", "image_url", "link", "category", "rating", "availability"]
                        st.warning("Auto field detection failed. Using common fallback fields.")
                else:
                    actual_fields = st.session_state['fields']
                
                # Run automated pagination scraping
                try:
                    st.info("🤖 Starting automated pagination detection...")
                    
                    # First, test pagination detection on the initial page
                    test_markdown = fetch_html_selenium(st.session_state['urls'][0], attended_mode=False)
                    test_markdown = html_to_markdown_with_readability(test_markdown)
                    
                    # Quick pagination test
                    pagination_test, test_tokens = detect_pagination_elements(
                        st.session_state['urls'][0], 
                        st.session_state['pagination_details'], 
                        st.session_state['model_selection'], 
                        test_markdown
                    )
                    
                    # Check if pagination was detected
                    if isinstance(pagination_test, dict):
                        test_urls = pagination_test.get("page_urls", [])
                    else:
                        test_urls = pagination_test.page_urls if pagination_test else []
                    
                    if test_urls:
                        st.success(f"🎯 Pagination detected! Found {len(test_urls)} potential pages to scrape.")
                        with st.expander("Preview detected URLs"):
                            for i, url in enumerate(test_urls[:10], 1):  # Show first 10
                                st.write(f"{i}. {url}")
                            if len(test_urls) > 10:
                                st.write(f"... and {len(test_urls) - 10} more URLs")
                    else:
                        st.warning("⚠️ No pagination URLs detected. Will scrape only the current page.")
                        st.info("💡 Try adding pagination hints like 'Look for numbered pages' or 'Find Next button'")
                    
                    # Proceed with full automated scraping
                    automation_results = run_automated_pagination_scrape(
                        base_url=st.session_state['urls'][0],
                        fields=actual_fields,
                        selected_model=st.session_state['model_selection'],
                        pagination_indications=st.session_state['pagination_details'],
                        output_folder=output_folder,
                        max_pages=st.session_state['max_pages'],
                        start_page=st.session_state.get('start_page', 1),
                        end_page=st.session_state.get('end_page', None)
                    )
                    
                    if automation_results['success']:
                        # Extract results from automation
                        all_data = []  # Will be populated from saved files
                        
                        # Create pagination info for display
                        pagination_info = {
                            "page_urls": automation_results['scraped_urls'],
                            "total_pages": automation_results['total_pages_scraped'],
                            "total_listings": automation_results['total_listings_extracted']
                        }
                        
                        # Load the combined data for display
                        combined_file = os.path.join(output_folder, 'combined_all_pages.json')
                        if os.path.exists(combined_file):
                            with open(combined_file, 'r', encoding='utf-8') as f:
                                combined_data = json.load(f)
                                all_data.append(combined_data)
                        
                        st.success(f"🎉 Automated pagination completed! Scraped {automation_results['total_pages_scraped']} pages with {automation_results['total_listings_extracted']} total listings.")
                        
                    else:
                        error_msg = automation_results.get('error', 'Unknown error')
                        st.error(f"❌ Automated pagination failed: {error_msg}")
                        
                        # Show debugging information
                        pages_attempted = automation_results.get('total_pages_scraped', 0)
                        if pages_attempted > 0:
                            st.warning(f"⚠️ Partial success: {pages_attempted} pages were scraped before the error occurred.")
                            
                        st.info("🔧 **Troubleshooting suggestions:**")
                        st.write("1. **Check pagination hints**: Try adding specific hints like 'Look for numbered pages 1,2,3' or 'Find Next button'")
                        st.write("2. **Website structure**: Some sites use infinite scroll or AJAX loading instead of traditional pagination")
                        st.write("3. **Anti-bot measures**: The website might be blocking automated requests")
                        st.write("4. **Try manual mode**: Disable automated pagination and check 'Enable Pagination' for manual URL detection")
                        
                        # Fallback to single page scraping
                        st.info("🔄 Falling back to single page scraping...")
                        # Fallback to manual mode
                        st.session_state['auto_paginate'] = False
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Automated pagination error: {str(e)}")
                    # Fallback to single page scraping
                    url = st.session_state['urls'][0]
                    raw_html = fetch_html_selenium(url, attended_mode=False)
                    markdown = html_to_markdown_with_readability(raw_html)
                    save_raw_data(markdown, output_folder, f'rawData_1.md')
                    
                    if show_tags:
                        actual_fields = st.session_state['fields']
                        DynamicListingModel = create_dynamic_listing_model(actual_fields)
                        DynamicListingsContainer = create_listings_container_model(DynamicListingModel)
                        
                        try:
                            formatted_data, token_counts = format_data(
                                markdown, DynamicListingsContainer, DynamicListingModel, st.session_state['model_selection']
                            )
                            save_formatted_data(formatted_data, output_folder, f'sorted_data_1.json', f'sorted_data_1.xlsx')
                            all_data.append(formatted_data)
                            st.success("✅ Fallback scraping completed successfully!")
                        except Exception as e:
                            st.error(f"❌ Error during fallback scraping: {str(e)}")
                            all_data.append({"error": str(e), "listings": []})
            
            else:
                # Original non-automated pagination logic
                for i, url in enumerate(st.session_state['urls'], start=1):
                    # Fetch HTML
                    raw_html = fetch_html_selenium(url, attended_mode=False)
                    markdown = html_to_markdown_with_readability(raw_html)
                    save_raw_data(markdown, output_folder, f'rawData_{i}.md')

                    # Detect pagination if enabled and only for the first URL (manual mode)
                    if st.session_state['use_pagination'] and i == 1 and not st.session_state.get('auto_paginate', False):
                        try:
                            pagination_data, token_counts = detect_pagination_elements(
                                url, st.session_state['pagination_details'], st.session_state['model_selection'], markdown
                            )
                            # Check if pagination_data is a dict or a model with 'page_urls' attribute
                            if isinstance(pagination_data, dict):
                                page_urls = pagination_data.get("page_urls", [])
                            else:
                                page_urls = pagination_data.page_urls if pagination_data else []
                            
                            pagination_info = {
                                "page_urls": page_urls,
                                "token_counts": token_counts
                            }
                        except Exception as e:
                            st.warning(f"Pagination detection failed: {str(e)}")
                            pagination_info = None
                        
                # Scrape data if fields are specified
                if show_tags:
                    # Handle extract all fields mode
                    if st.session_state['extract_all_fields']:
                        # Get all available fields from the content
                        all_fields = extract_all_fields_from_content(url, st.session_state['gemini_api_key'])
                        if all_fields:
                            actual_fields = all_fields
                            st.info(f"🎯 URL {i}: Extracted {len(all_fields)} fields: {', '.join(all_fields[:10])}{'...' if len(all_fields) > 10 else ''}")
                        else:
                            # Fallback to common fields if auto-detection fails
                            actual_fields = ["title", "description", "price", "image_url", "link", "category", "rating", "availability"]
                            st.warning(f"URL {i}: Auto field detection failed. Using common fallback fields.")
                    else:
                        actual_fields = st.session_state['fields']
                    
                    # Create dynamic models
                    DynamicListingModel = create_dynamic_listing_model(actual_fields)
                    DynamicListingsContainer = create_listings_container_model(DynamicListingModel)
                    
                    # Format data with error handling
                    try:
                        formatted_data, token_counts = format_data(
                            markdown, DynamicListingsContainer, DynamicListingModel, st.session_state['model_selection']
                        )
                        # Save formatted data
                        df = save_formatted_data(formatted_data, output_folder, f'sorted_data_{i}.json', f'sorted_data_{i}.xlsx')
                        all_data.append(formatted_data)
                        if i == 1:  # Only show success message for first URL
                            st.success(f"✅ URL {i}: Data extracted successfully!")
                    except Exception as e:
                        st.error(f"❌ URL {i}: Error during data extraction: {str(e)}")
                        all_data.append({"error": str(e), "listings": []})

        # Clean up driver if used
        if driver:
            driver.quit()
            st.session_state['driver'] = None

        # Save results
        st.session_state['results'] = {
            'data': all_data,
            'output_folder': output_folder,
            'pagination_info': pagination_info
        }
        st.session_state['scraping_state'] = 'completed'

# Display results
if st.session_state['scraping_state'] == 'completed' and st.session_state['results']:
    results = st.session_state['results']
    all_data = results['data']
    output_folder = results['output_folder']
    pagination_info = results['pagination_info']

    # Display scraping details
    if show_tags:
        st.subheader("Scraping Results")
        
        # Show extraction mode used
        if st.session_state.get('extract_all_fields', False):
            st.success("🎯 **Extract All Fields Mode** was used - automatically detected and extracted all available structured data fields.")
        
        for i, data in enumerate(all_data, start=1):
            st.write(f"Data from URL {i}:")
            
            # Handle string data (convert to dict if it's JSON)
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    st.error(f"Failed to parse data as JSON for URL {i}")
                    continue
            
            if isinstance(data, dict):
                if 'listings' in data and isinstance(data['listings'], list):
                    df = pd.DataFrame(data['listings'])
                else:
                    # If 'listings' is not in the dict or not a list, use the entire dict
                    df = pd.DataFrame([data])
            elif hasattr(data, 'listings') and isinstance(data.listings, list):
                # Handle the case where data is a Pydantic model
                listings = [item.dict() for item in data.listings]
                df = pd.DataFrame(listings)
            else:
                st.error(f"Unexpected data format for URL {i}")
                continue
            
            # Show field count for extract all mode
            if st.session_state.get('extract_all_fields', False) and not df.empty:
                st.info(f"📊 Extracted {len(df.columns)} fields: {', '.join(list(df.columns)[:10])}{'...' if len(df.columns) > 10 else ''}")
            
            # Display the dataframe
            st.dataframe(df, use_container_width=True)

        # Download options
        st.subheader("Download Extracted Data")
        col1, col2 = st.columns(2)
        with col1:
            json_data = json.dumps(all_data, default=lambda o: o.dict() if hasattr(o, 'dict') else str(o), indent=4)
            st.download_button(
                "Download JSON",
                data=json_data,
                file_name="scraped_data.json"
            )
        with col2:
            # Convert all data to a single DataFrame
            all_listings = []
            for data in all_data:
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                if isinstance(data, dict) and 'listings' in data:
                    all_listings.extend(data['listings'])
                elif hasattr(data, 'listings'):
                    all_listings.extend([item.dict() for item in data.listings])
                else:
                    all_listings.append(data)
            
            combined_df = pd.DataFrame(all_listings)
            st.download_button(
                "Download CSV",
                data=combined_df.to_csv(index=False),
                file_name="scraped_data.csv"
            )

        st.success(f"Scraping completed. Results saved in {output_folder}")
        
        # Show processing summary
        st.subheader("� Processing Summary")
        if pagination_info and pagination_info.get("total_pages"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Pages Processed", pagination_info["total_pages"])
            with col2:
                st.metric("Total Listings", pagination_info.get("total_listings", "N/A"))
        else:
            st.info("Single page processing completed successfully.")

    # Display pagination results only if pagination_info exists and has valid data
    if pagination_info and pagination_info.get("page_urls"):
        st.subheader("Pagination Results")
        
        # Check if this is from automated pagination
        if pagination_info.get("total_pages") and pagination_info.get("total_listings"):
            # Automated pagination results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Pages Scraped", pagination_info["total_pages"])
            with col2:
                st.metric("Total Listings", pagination_info["total_listings"])
            
            st.success(f"🤖 **Automated Pagination Completed Successfully!**")
            st.info(f"📊 All {pagination_info['total_pages']} pages were automatically discovered and scraped.")
        
        # Display page URLs in a table
        st.write("**Scraped Page URLs:**")
        # Make URLs clickable
        pagination_df = pd.DataFrame(pagination_info["page_urls"], columns=["Page URLs"])
        
        st.dataframe(
            pagination_df,
            column_config={
                "Page URLs": st.column_config.LinkColumn("Page URLs")
            },
            use_container_width=True
        )

        # Download pagination URLs
        st.subheader("Download Pagination URLs")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download Pagination CSV",
                data=pagination_df.to_csv(index=False),
                file_name="pagination_urls.csv"
            )
        with col2:
            st.download_button(
                "Download Pagination JSON",
                data=json.dumps(pagination_info['page_urls'], indent=4),
                file_name="pagination_urls.json"
            )
    elif st.session_state.get('use_pagination', False):
        if st.session_state.get('auto_paginate', False):
            st.info("Automated pagination was enabled but no additional pages were found.")
        else:
            st.info("Pagination was enabled but no pagination URLs were found.")

    # Reset scraping state
    if st.sidebar.button("Clear Results"):
        st.session_state['scraping_state'] = 'idle'
        st.session_state['results'] = None
        st.session_state['suggested_fields'] = []
        st.session_state['search_results'] = None

# Helper function to generate unique folder names
def generate_unique_folder_name(url):
    timestamp = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    
    # Parse the URL
    parsed_url = urlparse(url)
    
    # Extract the domain name
    domain = parsed_url.netloc or parsed_url.path.split('/')[0]
    
    # Remove 'www.' if present
    domain = re.sub(r'^www\.', '', domain)
    
    # Remove any non-alphanumeric characters and replace with underscores
    clean_domain = re.sub(r'\W+', '_', domain)
    
    return f"{clean_domain}_{timestamp}"
