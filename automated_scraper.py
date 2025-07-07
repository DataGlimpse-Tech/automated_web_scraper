# automated_scraper.py

import os
import json
import logging
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import pandas as pd

from scraper import (
    fetch_html_selenium,
    html_to_markdown_with_readability,
    create_dynamic_listing_model,
    create_listings_container_model,
    format_data,
    save_raw_data,
    save_formatted_data,
    generate_unique_folder_name
)
from pagination_detector import (
    scrape_all_paginated_urls,
    calculate_total_pagination_stats,
    get_smart_pagination_settings
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AutomatedPaginationScraper:
    """
    Automated scraper that handles pagination without manual intervention.
    """
    
    def __init__(self, base_url: str, fields: List[str], selected_model: str, 
                 output_folder: Optional[str] = None, pagination_indications: str = "",
                 max_pages: int = 5, start_page: int = 1, end_page: Optional[int] = None):
        """
        Initialize the automated pagination scraper.
        
        Args:
            base_url: The starting URL to scrape
            fields: List of fields to extract from each page
            selected_model: AI model to use for data extraction and pagination detection
            output_folder: Custom output folder (if None, will generate unique folder)
            pagination_indications: User hints for pagination detection
            max_pages: Maximum number of pages to scrape (default: 5)
            start_page: Which page to start from (default: 1)
            end_page: Which page to end at (if None, uses start_page + max_pages - 1)
        """
        self.base_url = base_url
        self.fields = fields
        self.selected_model = selected_model
        self.pagination_indications = pagination_indications
        self.output_folder = output_folder or f"output/{generate_unique_folder_name(base_url)}"
        
        # Page limits
        self.max_pages = max_pages
        self.start_page = start_page
        self.end_page = end_page or (start_page + max_pages - 1)
        
        # Get smart pagination settings based on domain
        self.pagination_settings = get_smart_pagination_settings(base_url)
        # Override max_pages with user setting
        self.pagination_settings['max_pages'] = self.max_pages
        
        # Initialize results storage
        self.scraped_pages = []
        self.extracted_data = []
        self.total_stats = {
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'pages_processed': 0
        }
        
        logging.info(f"Initialized AutomatedPaginationScraper for {base_url}")
        logging.info(f"Page limits: {self.start_page}-{self.end_page} (max {self.max_pages} pages)")
        logging.info(f"Pagination settings: {self.pagination_settings}")
    
    def run_complete_scrape(self) -> Dict:
        """
        Run the complete automated scraping process with pagination.
        
        Returns:
            Dictionary containing all results, processing stats, and metadata
        """
        try:
            logging.info("Starting complete automated scraping process...")
            
            # Step 1: Get initial page content
            initial_markdown = self._get_initial_page_content()
            if not initial_markdown:
                raise Exception("Failed to fetch initial page content")
            
            # Step 2: Perform automated pagination scraping
            logging.info(f"Calling scrape_all_paginated_urls with:")
            logging.info(f"  base_url: {self.base_url}")
            logging.info(f"  max_pages: {self.pagination_settings['max_pages']}")
            logging.info(f"  start_page: {self.start_page}")
            logging.info(f"  end_page: {self.end_page}")
            logging.info(f"  delay_range: {self.pagination_settings['delay_range']}")
            
            self.scraped_pages = scrape_all_paginated_urls(
                base_url=self.base_url,
                indications=self.pagination_indications,
                selected_model=self.selected_model,
                initial_markdown=initial_markdown,
                max_pages=self.pagination_settings['max_pages'],
                delay_range=self.pagination_settings['delay_range'],
                start_page=self.start_page,
                end_page=self.end_page
            )
            
            if not self.scraped_pages:
                logging.warning("No pages were scraped by automated pagination")
                # Add the initial page as a fallback
                self.scraped_pages = [{
                    'page_number': 1,
                    'url': self.base_url,
                    'markdown_content': initial_markdown,
                    'pagination_data': {'page_urls': []},
                    'token_counts': {'input_tokens': 0, 'output_tokens': 0},
                    'pagination_stats': {},
                    'timestamp': time.time()
                }]
                logging.info("Added initial page as fallback")
            
            logging.info(f"Total pages to process: {len(self.scraped_pages)}")
            
            # Step 3: Calculate pagination statistics
            pagination_stats = calculate_total_pagination_stats(self.scraped_pages)
            self.total_stats.update({
                'total_input_tokens': pagination_stats['total_input_tokens'],
                'total_output_tokens': pagination_stats['total_output_tokens'],
                'pages_processed': pagination_stats['pages_processed']
            })
            
            # Step 4: Extract data from all pages
            self._extract_data_from_all_pages()
            
            # Step 5: Save all results
            self._save_all_results()
            
            # Step 6: Generate summary
            summary = self._generate_summary()
            
            logging.info("Complete automated scraping process finished successfully!")
            return summary
            
        except Exception as e:
            logging.error(f"Error in automated scraping process: {e}")
            return {
                'success': False,
                'error': str(e),
                'pages_scraped': len(self.scraped_pages),
                'stats': self.total_stats
            }
    
    def _get_initial_page_content(self) -> Optional[str]:
        """Get the markdown content of the initial page."""
        try:
            logging.info(f"Fetching initial page content from: {self.base_url}")
            raw_html = fetch_html_selenium(self.base_url, attended_mode=False)
            markdown = html_to_markdown_with_readability(raw_html)
            logging.info("Successfully fetched initial page content")
            return markdown
        except Exception as e:
            logging.error(f"Failed to fetch initial page content: {e}")
            return None
    
    def _extract_data_from_all_pages(self):
        """Extract structured data from all scraped pages."""
        logging.info(f"Extracting data from {len(self.scraped_pages)} pages...")
        
        # Create dynamic models
        DynamicListingModel = create_dynamic_listing_model(self.fields)
        DynamicListingsContainer = create_listings_container_model(DynamicListingModel)
        
        extraction_input_tokens = 0
        extraction_output_tokens = 0
        
        for i, page_data in enumerate(self.scraped_pages, 1):
            try:
                logging.info(f"Extracting data from page {i}/{len(self.scraped_pages)}")
                
                # Extract data from page
                formatted_data, token_counts = format_data(
                    page_data['markdown_content'],
                    DynamicListingsContainer,
                    DynamicListingModel,
                    self.selected_model
                )
                
                # Track token usage
                extraction_input_tokens += token_counts.get('input_tokens', 0)
                extraction_output_tokens += token_counts.get('output_tokens', 0)
                
                # Store extracted data with metadata
                page_result = {
                    'page_number': page_data['page_number'],
                    'url': page_data['url'],
                    'extracted_data': formatted_data,
                    'token_counts': token_counts,
                    'timestamp': page_data['timestamp']
                }
                self.extracted_data.append(page_result)
                
                logging.info(f"Successfully extracted data from page {i}")
                
                # Add small delay between extractions to avoid rate limits
                if i < len(self.scraped_pages):
                    time.sleep(1)
                    
            except Exception as e:
                logging.error(f"Failed to extract data from page {i}: {e}")
                # Continue with other pages even if one fails
                continue
        
        # Update total stats
        self.total_stats.update({
            'total_input_tokens': self.total_stats['total_input_tokens'] + extraction_input_tokens,
            'total_output_tokens': self.total_stats['total_output_tokens'] + extraction_output_tokens
        })
        
        logging.info(f"Data extraction completed. Processed {len(self.extracted_data)} pages successfully.")
    
    def _save_all_results(self):
        """Save all scraped data and metadata to files."""
        logging.info("Saving all results to files...")
        
        # Ensure output folder exists
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Save raw markdown data for each page
        for page_data in self.scraped_pages:
            page_num = page_data['page_number']
            raw_file = f'rawData_page_{page_num}.md'
            save_raw_data(page_data['markdown_content'], self.output_folder, raw_file)
        
        # Save extracted data for each page
        all_listings = []
        for page_result in self.extracted_data:
            page_num = page_result['page_number']
            
            # Save individual page data
            json_file = f'sorted_data_page_{page_num}.json'
            excel_file = f'sorted_data_page_{page_num}.xlsx'
            save_formatted_data(page_result['extracted_data'], self.output_folder, json_file, excel_file)
            
            # Collect all listings for combined file
            formatted_data = page_result['extracted_data']
            if isinstance(formatted_data, str):
                try:
                    formatted_data = json.loads(formatted_data)
                except json.JSONDecodeError:
                    continue
            
            if isinstance(formatted_data, dict) and 'listings' in formatted_data:
                for listing in formatted_data['listings']:
                    listing['source_page'] = page_num
                    listing['source_url'] = page_result['url']
                    all_listings.append(listing)
            elif hasattr(formatted_data, 'listings'):
                for listing in formatted_data.listings:
                    listing_dict = listing.dict() if hasattr(listing, 'dict') else listing
                    listing_dict['source_page'] = page_num
                    listing_dict['source_url'] = page_result['url']
                    all_listings.append(listing_dict)
        
        # Save combined data
        if all_listings:
            combined_data = {'listings': all_listings}
            save_formatted_data(combined_data, self.output_folder, 'combined_all_pages.json', 'combined_all_pages.xlsx')
        
        # Save pagination metadata
        pagination_metadata = {
            'base_url': self.base_url,
            'total_pages_scraped': len(self.scraped_pages),
            'pagination_settings': self.pagination_settings,
            'scraped_urls': [page['url'] for page in self.scraped_pages],
            'stats': self.total_stats,
            'fields_extracted': self.fields,
            'model_used': self.selected_model,
            'scraping_timestamp': datetime.now().isoformat()
        }
        
        metadata_file = os.path.join(self.output_folder, 'pagination_metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(pagination_metadata, f, indent=4)
        
        logging.info(f"All results saved to: {self.output_folder}")
    
    def _generate_summary(self) -> Dict:
        """Generate a summary of the scraping results."""
        total_listings = 0
        
        for page_result in self.extracted_data:
            formatted_data = page_result['extracted_data']
            if isinstance(formatted_data, str):
                try:
                    formatted_data = json.loads(formatted_data)
                except json.JSONDecodeError:
                    continue
            
            if isinstance(formatted_data, dict) and 'listings' in formatted_data:
                total_listings += len(formatted_data['listings'])
            elif hasattr(formatted_data, 'listings'):
                total_listings += len(formatted_data.listings)
        
        summary = {
            'success': True,
            'base_url': self.base_url,
            'total_pages_scraped': len(self.scraped_pages),
            'total_pages_with_data': len(self.extracted_data),
            'total_listings_extracted': total_listings,
            'fields_extracted': self.fields,
            'model_used': self.selected_model,
            'output_folder': self.output_folder,
            'stats': self.total_stats,
            'pagination_settings': self.pagination_settings,
            'scraped_urls': [page['url'] for page in self.scraped_pages],
            'scraping_duration': max([page['timestamp'] for page in self.scraped_pages]) - min([page['timestamp'] for page in self.scraped_pages]) if self.scraped_pages else 0
        }
        
        return summary

def run_automated_pagination_scrape(base_url: str, fields: List[str], selected_model: str, 
                                   pagination_indications: str = "", output_folder: Optional[str] = None,
                                   max_pages: int = 5, start_page: int = 1, end_page: Optional[int] = None) -> Dict:
    """
    Convenience function to run automated pagination scraping.
    
    Args:
        base_url: The starting URL to scrape
        fields: List of fields to extract from each page
        selected_model: AI model to use for data extraction and pagination detection
        pagination_indications: User hints for pagination detection
        output_folder: Custom output folder (if None, will generate unique folder)
        max_pages: Maximum number of pages to scrape (default: 5)
        start_page: Which page to start from (default: 1)
        end_page: Which page to end at (if None, uses start_page + max_pages - 1)
    
    Returns:
        Dictionary containing all results, processing stats, and metadata
    """
    scraper = AutomatedPaginationScraper(
        base_url=base_url,
        fields=fields,
        selected_model=selected_model,
        output_folder=output_folder,
        pagination_indications=pagination_indications,
        max_pages=max_pages,
        start_page=start_page,
        end_page=end_page
    )
    
    return scraper.run_complete_scrape()
