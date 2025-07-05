#!/usr/bin/env python3
"""
Debug script to test and troubleshoot automated pagination issues.
"""

import os
import json
import logging
from datetime import datetime
from scraper import fetch_html_selenium, html_to_markdown_with_readability
from pagination_detector import detect_pagination_elements, scrape_all_paginated_urls
import google.generativeai as genai

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def debug_pagination_step_by_step(url: str, gemini_api_key: str, max_pages: int = 3):
    """
    Debug pagination detection step by step to identify issues.
    """
    print("🔍 DEBUG: Starting Pagination Debugging")
    print(f"URL: {url}")
    print(f"Max Pages: {max_pages}")
    print("-" * 60)
    
    # Step 1: Test basic HTML fetching
    print("Step 1: Testing HTML fetching...")
    try:
        raw_html = fetch_html_selenium(url, attended_mode=False)
        print(f"✅ HTML fetched successfully. Length: {len(raw_html)} characters")
        
        # Save raw HTML for inspection
        debug_folder = f"debug_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(debug_folder, exist_ok=True)
        
        with open(f"{debug_folder}/raw_html.html", 'w', encoding='utf-8') as f:
            f.write(raw_html)
        print(f"💾 Raw HTML saved to {debug_folder}/raw_html.html")
        
    except Exception as e:
        print(f"❌ Error fetching HTML: {e}")
        return
    
    # Step 2: Test markdown conversion
    print("\nStep 2: Testing markdown conversion...")
    try:
        markdown_content = html_to_markdown_with_readability(raw_html)
        print(f"✅ Markdown converted successfully. Length: {len(markdown_content)} characters")
        
        # Save markdown for inspection
        with open(f"{debug_folder}/markdown_content.md", 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        print(f"💾 Markdown saved to {debug_folder}/markdown_content.md")
        
        # Show first 500 characters
        print("\n📄 First 500 characters of markdown:")
        print("-" * 40)
        print(markdown_content[:500])
        print("-" * 40)
        
    except Exception as e:
        print(f"❌ Error converting to markdown: {e}")
        return
    
    # Step 3: Test pagination detection
    print("\nStep 3: Testing pagination detection...")
    try:
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        
        pagination_data, token_counts = detect_pagination_elements(
            url, "", "gemini-1.5-flash", markdown_content
        )
        
        print(f"✅ Pagination detection completed")
        print(f"📊 Token usage: {token_counts}")
        
        # Show pagination results
        if isinstance(pagination_data, dict):
            page_urls = pagination_data.get("page_urls", [])
        else:
            page_urls = pagination_data.page_urls if pagination_data else []
        
        print(f"🔗 Found {len(page_urls)} pagination URLs:")
        for i, page_url in enumerate(page_urls, 1):
            print(f"  {i}. {page_url}")
        
        # Save pagination results
        pagination_results = {
            "base_url": url,
            "detected_urls": page_urls,
            "token_counts": token_counts,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(f"{debug_folder}/pagination_results.json", 'w', encoding='utf-8') as f:
            json.dump(pagination_results, f, indent=4)
        
        print(f"💾 Pagination results saved to {debug_folder}/pagination_results.json")
        
    except Exception as e:
        print(f"❌ Error in pagination detection: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Test automated scraping (if URLs were found)
    if page_urls:
        print(f"\nStep 4: Testing automated scraping (max {max_pages} pages)...")
        try:
            scraped_pages = scrape_all_paginated_urls(
                base_url=url,
                indications="",
                selected_model="gemini-1.5-flash",
                initial_markdown=markdown_content,
                max_pages=max_pages,  # Use the provided limit
                delay_range=(1, 2),  # Shorter delay for debugging
                start_page=1,
                end_page=max_pages
            )
            
            print(f"✅ Automated scraping completed")
            print(f"📄 Scraped {len(scraped_pages)} pages total")
            
            # Show scraped page info
            for i, page_data in enumerate(scraped_pages, 1):
                print(f"  Page {i}: {page_data['url']} (Content length: {len(page_data['markdown_content'])} chars)")
            
            # Save scraping results
            scraping_results = {
                "total_pages": len(scraped_pages),
                "scraped_urls": [page['url'] for page in scraped_pages],
                "page_details": [
                    {
                        "page_number": page['page_number'],
                        "url": page['url'],
                        "content_length": len(page['markdown_content']),
                        "pagination_urls_found": len(page['pagination_data'].page_urls if hasattr(page['pagination_data'], 'page_urls') else page['pagination_data'].get('page_urls', []))
                    }
                    for page in scraped_pages
                ]
            }
            
            with open(f"{debug_folder}/scraping_results.json", 'w', encoding='utf-8') as f:
                json.dump(scraping_results, f, indent=4)
                
            print(f"💾 Scraping results saved to {debug_folder}/scraping_results.json")
            
        except Exception as e:
            print(f"❌ Error in automated scraping: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠️  No pagination URLs found, skipping automated scraping test")
    
    print(f"\n🎯 Debug session completed. Results saved in: {debug_folder}")
    print("\n💡 Next steps:")
    print("1. Check the raw HTML file to see if the page loaded correctly")
    print("2. Check the markdown content to see if pagination elements are visible")
    print("3. Review the pagination results to see what URLs were detected")
    print("4. If no URLs were found, the AI model might need better prompting or the site has different pagination")

def debug_specific_website_pagination(url: str, gemini_api_key: str, user_hints: str = ""):
    """
    Debug pagination for a specific website with custom hints.
    """
    print("🎯 DEBUG: Testing pagination with custom hints")
    print(f"URL: {url}")
    print(f"Hints: {user_hints}")
    print("-" * 60)
    
    try:
        # Get content
        raw_html = fetch_html_selenium(url, attended_mode=False)
        markdown_content = html_to_markdown_with_readability(raw_html)
        
        # Test with custom hints
        genai.configure(api_key=gemini_api_key)
        
        pagination_data, token_counts = detect_pagination_elements(
            url, user_hints, "gemini-1.5-flash", markdown_content
        )
        
        if isinstance(pagination_data, dict):
            page_urls = pagination_data.get("page_urls", [])
        else:
            page_urls = pagination_data.page_urls if pagination_data else []
        
        print(f"🔗 Found {len(page_urls)} pagination URLs with hints:")
        for i, page_url in enumerate(page_urls, 1):
            print(f"  {i}. {page_url}")
            
        return page_urls
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

if __name__ == "__main__":
    # Example usage
    TEST_URL = "https://builtin.com/companies"
    
    # You'll need to set your Gemini API key
    GEMINI_API_KEY = input("Enter your Gemini API key: ")
    MAX_PAGES = int(input("Enter max pages to test (default 3): ") or "3")
    
    if GEMINI_API_KEY:
        # Run basic debug
        debug_pagination_step_by_step(TEST_URL, GEMINI_API_KEY, MAX_PAGES)
        
        # Test with specific hints for builtin.com
        print("\n" + "="*80)
        hints = "Look for numbered pages at the bottom, 'Next' buttons, or 'Load More' buttons. Check for URLs with page parameters like ?page=2"
        debug_specific_website_pagination(TEST_URL, GEMINI_API_KEY, hints)
    else:
        print("❌ Gemini API key required for debugging")
