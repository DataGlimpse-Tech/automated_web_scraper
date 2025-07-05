# pagination_detector.py

import os
import json
from typing import List, Dict, Tuple, Union, Optional
from pydantic import BaseModel, Field, ValidationError

from dotenv import load_dotenv

import google.generativeai as genai

from api_management import get_api_key
from assets import PROMPT_PAGINATION

load_dotenv()
import logging
import time
import random
from urllib.parse import urljoin, urlparse

class PaginationData(BaseModel):
    page_urls: List[str] = Field(default_factory=list, description="List of pagination URLs, including 'Next' button URL if present")

def detect_pagination_elements(url: str, indications: str, selected_model: str, markdown_content: str) -> Tuple[Union[PaginationData, Dict, str], Dict]:
    try:
        """
        Uses Gemini AI model to analyze markdown content and extract pagination elements.

        Args:
            url (str): The URL of the page to extract pagination from.
            indications (str): User indications for pagination detection.
            selected_model (str): The name of the model to use (should be "gemini-1.5-flash").
            markdown_content (str): The markdown content to analyze.

        Returns:
            Tuple[PaginationData, Dict]: Parsed pagination data and token counts.
        """ 
        prompt_pagination = PROMPT_PAGINATION+"\n The url of the page to extract pagination from: "+url+"\nIf the urls that you find are not complete combine them intelligently in a way that fit the pattern **ALWAYS GIVE A FULL URL**"
        if indications != "":
            prompt_pagination +="\n\n These are the users indications that, pay special attention to them: "+indications+"\n\n Below are the markdowns of the website: \n\n"
        else:
            prompt_pagination +="\n There are no user indications in this case just apply the logic described. \n\n Below are the markdowns of the website: \n\n"

        if selected_model == "gemini-1.5-flash":
            # Use Google Gemini API
            genai.configure(api_key=get_api_key("GOOGLE_API_KEY"))
            model = genai.GenerativeModel(
                'gemini-1.5-flash',
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": PaginationData
                }
            )
            prompt = f"{prompt_pagination}\n{markdown_content}"
            # Count input tokens using Gemini's method
            input_tokens = model.count_tokens(prompt)
            completion = model.generate_content(prompt)
            # Extract token counts from usage_metadata
            usage_metadata = completion.usage_metadata
            token_counts = {
                "input_tokens": usage_metadata.prompt_token_count,
                "output_tokens": usage_metadata.candidates_token_count
            }
            # Get the result
            response_content = completion.text
            
            # Log the response content for debugging
            logging.info(f"Gemini Flash response content: {response_content[:500]}...")
            
            # Try to parse the response as JSON
            try:
                parsed_data = json.loads(response_content)
                if isinstance(parsed_data, dict) and 'page_urls' in parsed_data:
                    # Filter out empty URLs and validate URLs
                    valid_urls = []
                    for url_candidate in parsed_data['page_urls']:
                        if url_candidate and isinstance(url_candidate, str) and url_candidate.strip():
                            # Convert relative URLs to absolute
                            if not url_candidate.startswith(('http://', 'https://')):
                                url_candidate = urljoin(url, url_candidate)
                            valid_urls.append(url_candidate.strip())
                    
                    pagination_data = PaginationData(page_urls=valid_urls)
                    logging.info(f"Successfully parsed {len(valid_urls)} valid pagination URLs")
                else:
                    logging.warning("Response missing 'page_urls' field")
                    pagination_data = PaginationData(page_urls=[])
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse Gemini Flash response as JSON: {e}")
                logging.error(f"Response content: {response_content}")
                pagination_data = PaginationData(page_urls=[])

            return pagination_data, token_counts
        else:
            raise ValueError(f"Unsupported model: {selected_model}. Only 'gemini-1.5-flash' is supported.")

    except Exception as e:
        logging.error(f"An error occurred in detect_pagination_elements: {e}")
        # Return default values if an error occurs
        return PaginationData(page_urls=[]), {"input_tokens": 0, "output_tokens": 0}


def scrape_all_paginated_urls(base_url: str, indications: str, selected_model: str, initial_markdown: str, 
                             max_pages: int = 5, delay_range: tuple = (2, 5), 
                             start_page: int = 1, end_page: Optional[int] = None) -> List[Dict]:
    """
    Automatically scrape all paginated URLs starting from the initial page.
    
    Args:
        base_url: The initial URL to start pagination from
        indications: User indications for pagination detection
        selected_model: AI model to use for pagination detection
        initial_markdown: Markdown content of the first page
        max_pages: Maximum number of pages to scrape (safety limit)
        delay_range: Random delay range between page requests (min, max) in seconds
        start_page: Page number to start from (1-based)
        end_page: Page number to end at (if None, uses start_page + max_pages - 1)
    
    Returns:
        List of dictionaries containing page data and metadata
    """
    logging.info(f"Starting automated pagination scraping for: {base_url}")
    logging.info(f"Page limits: {start_page} to {end_page or (start_page + max_pages - 1)} (max {max_pages} pages)")
    
    # Calculate actual end page
    if end_page is None:
        end_page = start_page + max_pages - 1
    
    scraped_pages = []
    current_page_num = start_page  # Start from specified page
    target_pages_to_scrape = end_page - start_page + 1
    processed_urls = set()  # Track processed URLs to avoid duplicates
    current_markdown = initial_markdown
    current_url = base_url
    
    # Add initial page to processed URLs
    processed_urls.add(current_url)
    
    # If we need to start from a page other than 1, we need to find the right starting URL
    if start_page > 1:
        # First detect pagination to find the starting page URL
        try:
            pagination_data, _, _ = detect_pagination_elements(
                current_url, indications, selected_model, current_markdown
            )
            
            if isinstance(pagination_data, dict):
                page_urls = pagination_data.get("page_urls", [])
            else:
                page_urls = pagination_data.page_urls if pagination_data else []
            
            # Look for URL with the start page number
            start_url = None
            for url in page_urls:
                import re
                page_matches = re.findall(r'(?:page[=\/]?|p[=\/]?)(\d+)', url.lower())
                if page_matches and int(page_matches[-1]) == start_page:
                    start_url = url
                    break
            
            if start_url:
                current_url = start_url
                # Fetch content of the starting page
                raw_html = fetch_html_selenium(start_url, attended_mode=False)
                current_markdown = html_to_markdown_with_readability(raw_html)
                logging.info(f"Starting from page {start_page}: {start_url}")
            else:
                logging.warning(f"Could not find URL for page {start_page}, starting from page 1")
                current_page_num = 1
                
        except Exception as e:
            logging.error(f"Error finding start page URL: {e}, starting from page 1")
            current_page_num = 1
    
    pages_scraped = 0  # Track how many pages we've actually scraped
    
    while pages_scraped < target_pages_to_scrape:
        logging.info(f"Processing page {current_page_num}: {current_url}")
        
        try:
            # Detect pagination on current page
            pagination_data, token_counts = detect_pagination_elements(
                current_url, indications, selected_model, current_markdown
            )
            
            # Store current page data
            page_data = {
                'page_number': current_page_num,
                'url': current_url,
                'markdown_content': current_markdown,
                'pagination_data': pagination_data,
                'token_counts': token_counts,
                'timestamp': time.time()
            }
            scraped_pages.append(page_data)
            pages_scraped += 1
            
            # Check if we've scraped enough pages
            if pages_scraped >= target_pages_to_scrape:
                logging.info(f"Reached target of {target_pages_to_scrape} pages. Stopping.")
                break
            
            # Get pagination URLs
            if isinstance(pagination_data, dict):
                page_urls = pagination_data.get("page_urls", [])
            else:
                page_urls = pagination_data.page_urls if pagination_data else []
            
            # Find next page URL that hasn't been processed
            next_url = None
            
            # Strategy 1: Look for URLs that contain page numbers higher than current page
            for url in page_urls:
                absolute_url = urljoin(current_url, url) if not url.startswith(('http://', 'https://')) else url
                
                if absolute_url not in processed_urls:
                    # Check if this URL likely represents the next page
                    if current_page_num == 1:  # For first page, accept any new URL
                        next_url = absolute_url
                        break
                    else:
                        # For subsequent pages, try to find URLs with incrementing numbers
                        import re
                        current_page_matches = re.findall(r'(?:page[=\/]?|p[=\/]?)(\d+)', current_url.lower())
                        next_page_matches = re.findall(r'(?:page[=\/]?|p[=\/]?)(\d+)', absolute_url.lower())
                        
                        if current_page_matches and next_page_matches:
                            current_num = int(current_page_matches[-1])
                            next_num = int(next_page_matches[-1])
                            if next_num > current_num:
                                next_url = absolute_url
                                break
                        else:
                            # If no clear pattern, accept the first unprocessed URL
                            next_url = absolute_url
                            break
            
            # Strategy 2: If no URL found, look for common pagination patterns
            if not next_url:
                next_patterns = ['next', 'siguiente', 'suivant', 'weiter', 'avanti', '次', '다음', '下一', 'more', 'load']
                for url in page_urls:
                    url_lower = url.lower()
                    if any(pattern in url_lower for pattern in next_patterns):
                        absolute_url = urljoin(current_url, url) if not url.startswith(('http://', 'https://')) else url
                        if absolute_url not in processed_urls:
                            next_url = absolute_url
                            break
            
            if not next_url:
                logging.info(f"No more pagination URLs found after page {current_page_num}. Total pages scraped: {pages_scraped}")
                logging.info(f"URLs found on last page: {page_urls}")
                logging.info(f"Processed URLs so far: {list(processed_urls)}")
                break
            
            # Add delay between requests to be respectful
            delay = random.uniform(delay_range[0], delay_range[1])
            logging.info(f"Waiting {delay:.2f} seconds before next request...")
            time.sleep(delay)
            
            # Move to next page
            processed_urls.add(next_url)
            current_url = next_url
            current_page_num += 1
            
            # Fetch content of next page
            try:
                from scraper import fetch_html_selenium, html_to_markdown_with_readability
                logging.info(f"Fetching content from: {next_url}")
                raw_html = fetch_html_selenium(next_url, attended_mode=False)
                current_markdown = html_to_markdown_with_readability(raw_html)
                
                # Validate content quality
                if len(current_markdown.strip()) < 100:
                    logging.warning(f"Very short content from {next_url}, might be an error page")
                
                if "no results" in current_markdown.lower() or "not found" in current_markdown.lower():
                    logging.warning(f"Page {next_url} appears to show no results or error")
                
                logging.info(f"Successfully fetched content for page {current_page_num} ({len(current_markdown)} chars)")
            except Exception as e:
                logging.error(f"Failed to fetch content for {next_url}: {e}")
                break
                
        except Exception as e:
            logging.error(f"Error processing page {current_page_num}: {e}")
            break
    
    logging.info(f"Pagination scraping completed. Total pages scraped: {len(scraped_pages)}")
    return scraped_pages

def calculate_total_pagination_stats(scraped_pages: List[Dict]) -> Dict:
    """
    Calculate total statistics for all paginated pages.
    
    Args:
        scraped_pages: List of page data dictionaries
    
    Returns:
        Dictionary with total token counts and page stats
    """
    total_input_tokens = 0
    total_output_tokens = 0
    
    for page_data in scraped_pages:
        if 'token_counts' in page_data and page_data['token_counts']:
            total_input_tokens += page_data['token_counts'].get('input_tokens', 0)
            total_output_tokens += page_data['token_counts'].get('output_tokens', 0)
    
    return {
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'pages_processed': len(scraped_pages)
    }

def get_smart_pagination_settings(url: str) -> Dict:
    """
    Provide intelligent pagination settings based on the website domain.
    
    Args:
        url: The URL to analyze
    
    Returns:
        Dictionary with recommended pagination settings
    """
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()
    
    # Domain-specific settings
    domain_settings = {
        'default': {
            'max_pages': 20,
            'delay_range': (2, 4),
            'respect_robots': True
        },
        'e-commerce': {
            'max_pages': 50,
            'delay_range': (1, 3),
            'respect_robots': True
        },
        'social': {
            'max_pages': 10,
            'delay_range': (3, 6),
            'respect_robots': True
        },
        'news': {
            'max_pages': 30,
            'delay_range': (2, 4),
            'respect_robots': True
        }
    }
    
    # Categorize domain
    if any(term in domain for term in ['shop', 'store', 'buy', 'cart', 'product']):
        return domain_settings['e-commerce']
    elif any(term in domain for term in ['facebook', 'twitter', 'instagram', 'linkedin', 'reddit']):
        return domain_settings['social']
    elif any(term in domain for term in ['news', 'blog', 'article', 'post']):
        return domain_settings['news']
    else:
        return domain_settings['default']
