import os
import random
import time
import re
import json
from datetime import datetime
from typing import List, Dict, Type

import pandas as pd
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, create_model
import html2text

from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

import google.generativeai as genai

from api_management import get_api_key
from assets import HEADLESS_OPTIONS,USER_MESSAGE,HEADLESS_OPTIONS_DOCKER
load_dotenv()


# Set up the Chrome WebDriver options


def is_running_in_docker():
    """
    Detect if the app is running inside a Docker container.
    This checks if the '/proc/1/cgroup' file contains 'docker'.
    """
    try:
        with open("/proc/1/cgroup", "rt") as file:
            return "docker" in file.read()
    except Exception:
        return False

def setup_selenium(attended_mode=False):
    options = Options()
    service = Service(ChromeDriverManager().install())

    # Apply headless options based on whether the code is running in Docker
    if is_running_in_docker():
        # Running inside Docker, use Docker-specific headless options
        for option in HEADLESS_OPTIONS_DOCKER:
            options.add_argument(option)
    else:
        # Not running inside Docker, use the normal headless options
        for option in HEADLESS_OPTIONS:
            options.add_argument(option)

    # Initialize the WebDriver
    driver = webdriver.Chrome(service=service, options=options)
    return driver




def fetch_html_selenium(url, attended_mode=False, driver=None):
    if driver is None:
        driver = setup_selenium(attended_mode)
        should_quit = True
        if not attended_mode:
            driver.get(url)
    else:
        should_quit = False
        # Do not navigate to the URL if in attended mode and driver is already initialized
        if not attended_mode:
            driver.get(url)

    try:
        if not attended_mode:
            # Add more realistic actions like scrolling
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
            time.sleep(random.uniform(1.1, 1.8))
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight/1.2);")
            time.sleep(random.uniform(1.1, 1.8))
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight/1);")
            time.sleep(random.uniform(1.1, 1.8))
        # Get the page source from the current page
        html = driver.page_source
        return html
    finally:
        if should_quit:
            driver.quit()




def clean_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove headers and footers based on common HTML tags or classes
    for element in soup.find_all(['header', 'footer']):
        element.decompose()  # Remove these tags and their content

    return str(soup)


def html_to_markdown_with_readability(html_content):

    
    cleaned_html = clean_html(html_content)  
    
    # Convert to markdown
    markdown_converter = html2text.HTML2Text()
    markdown_converter.ignore_links = False
    markdown_content = markdown_converter.handle(cleaned_html)
    
    return markdown_content


    
def save_raw_data(raw_data: str, output_folder: str, file_name: str):
    """Save raw markdown data to the specified output folder."""
    os.makedirs(output_folder, exist_ok=True)
    raw_output_path = os.path.join(output_folder, file_name)
    with open(raw_output_path, 'w', encoding='utf-8') as f:
        f.write(raw_data)
    return raw_output_path


def create_dynamic_listing_model(field_names: List[str]) -> Type[BaseModel]:
    """
    Dynamically creates a Pydantic model based on provided fields.
    field_name is a list of names of the fields to extract from the markdown.
    """
    # Create field definitions using aliases for Field parameters
    field_definitions = {field: (str, ...) for field in field_names}
    # Dynamically create the model with all field
    return create_model('DynamicListingModel', **field_definitions)


def create_listings_container_model(listing_model: Type[BaseModel]) -> Type[BaseModel]:
    """
    Create a container model that holds a list of the given listing model.
    """
    return create_model('DynamicListingsContainer', listings=(List[listing_model], ...))




def generate_system_message(listing_model: BaseModel) -> str:
    """
    Dynamically generate a system message based on the fields in the provided listing model.
    """
    # Use the model_json_schema() method to introspect the Pydantic model
    schema_info = listing_model.model_json_schema()

    # Extract field descriptions from the schema
    field_descriptions = []
    for field_name, field_info in schema_info["properties"].items():
        # Get the field type from the schema info
        field_type = field_info["type"]
        field_descriptions.append(f'"{field_name}": "{field_type}"')

    # Create the JSON schema structure for the listings
    schema_structure = ",\n".join(field_descriptions)

    # Generate the system message dynamically
    system_message = f"""
    You are an intelligent text extraction and conversion assistant. Your task is to extract structured information 
                        from the given text and convert it into a pure JSON format. The JSON should contain only the structured data extracted from the text, 
                        with no additional commentary, explanations, or extraneous information. 
                        You could encounter cases where you can't find the data of the fields you have to extract or the data will be in a foreign language.
                        
                        CRITICAL: Your response must be ONLY valid JSON. Do not include any text before or after the JSON. 
                        Do not include markdown formatting, code blocks, or any explanatory text.
                        
                        Please process the following text and provide the output in pure JSON format with no words before or after the JSON:
    Please ensure the output strictly follows this schema:

    {{
        "listings": [
            {{
                {schema_structure}
            }}
        ]
    }} """

    return system_message



def format_data(data, DynamicListingsContainer, DynamicListingModel, selected_model):
    token_counts = {}
    
    if selected_model == "gemini-1.5-flash":
        # Use Google Gemini API
        # Dynamically generate the system message based on the schema
        sys_message = generate_system_message(DynamicListingModel)
        genai.configure(api_key=get_api_key("GOOGLE_API_KEY"))
        model = genai.GenerativeModel('gemini-1.5-flash',
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": DynamicListingsContainer
                })
        prompt = sys_message + "\n" + USER_MESSAGE + data
        # Count input tokens using Gemini's method
        input_tokens = model.count_tokens(prompt)
        completion = model.generate_content(prompt)
        # Extract token counts from usage_metadata
        usage_metadata = completion.usage_metadata
        token_counts = {
            "input_tokens": usage_metadata.prompt_token_count,
            "output_tokens": usage_metadata.candidates_token_count
        }
        
        # Get the response text
        response_text = completion.text
        
        # Basic validation to ensure we have a valid response
        if not response_text or not response_text.strip():
            raise ValueError("Empty response from Gemini API")
            
        # Log the first 200 characters for debugging
        print(f"Gemini response preview: {response_text[:200]}...")
        
        return response_text, token_counts
    else:
        raise ValueError(f"Unsupported model: {selected_model}. Only 'gemini-1.5-flash' is supported.")



def save_formatted_data(formatted_data, output_folder: str, json_file_name: str, excel_file_name: str):
    """Save formatted data as JSON and Excel in the specified output folder."""
    os.makedirs(output_folder, exist_ok=True)
    
    # Parse the formatted data if it's a JSON string (from Gemini API)
    if isinstance(formatted_data, str):
        try:
            formatted_data_dict = json.loads(formatted_data)
        except json.JSONDecodeError:
            # Try to extract JSON from the response (sometimes Gemini adds extra text)
            import re
            
            # Look for JSON pattern in the string
            json_match = re.search(r'\{.*\}', formatted_data, re.DOTALL)
            if json_match:
                try:
                    formatted_data_dict = json.loads(json_match.group())
                except json.JSONDecodeError:
                    # If still failing, create a fallback structure
                    formatted_data_dict = {
                        "error": "Failed to parse JSON response",
                        "raw_response": formatted_data,
                        "listings": []
                    }
                    print(f"Warning: Failed to parse JSON response. Raw response: {formatted_data[:500]}...")
            else:
                # No JSON found, create fallback structure
                formatted_data_dict = {
                    "error": "No JSON found in response",
                    "raw_response": formatted_data,
                    "listings": []
                }
                print(f"Warning: No JSON found in response. Raw response: {formatted_data[:500]}...")
    else:
        # Handle data from OpenAI or other sources
        formatted_data_dict = formatted_data.dict() if hasattr(formatted_data, 'dict') else formatted_data

    # Save the formatted data as JSON
    json_output_path = os.path.join(output_folder, json_file_name)
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_data_dict, f, indent=4)

    # Prepare data for DataFrame
    if isinstance(formatted_data_dict, dict):
        # If the data is a dictionary containing lists, assume these lists are records
        data_for_df = next(iter(formatted_data_dict.values())) if len(formatted_data_dict) == 1 else formatted_data_dict
    elif isinstance(formatted_data_dict, list):
        data_for_df = formatted_data_dict
    else:
        raise ValueError("Formatted data is neither a dictionary nor a list, cannot convert to DataFrame")

    # Create DataFrame
    try:
        df = pd.DataFrame(data_for_df)

        # Save the DataFrame to an Excel file
        excel_output_path = os.path.join(output_folder, excel_file_name)
        df.to_excel(excel_output_path, index=False)
        
        return df
    except Exception as e:
        return None

def generate_unique_folder_name(url):
    timestamp = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    url_name = re.sub(r'\W+', '_', url.split('//')[1].split('/')[0])  # Extract domain name and replace non-alphanumeric characters
    return f"{url_name}_{timestamp}"


def scrape_url(url: str, fields: List[str], selected_model: str, output_folder: str, file_number: int, markdown: str):
    """Scrape a single URL and save the results."""
    try:
        # Save raw data
        save_raw_data(markdown, output_folder, f'rawData_{file_number}.md')

        # Create the dynamic listing model
        DynamicListingModel = create_dynamic_listing_model(fields)

        # Create the container model that holds a list of the dynamic listing models
        DynamicListingsContainer = create_listings_container_model(DynamicListingModel)
        
        # Format data
        formatted_data, token_counts = format_data(markdown, DynamicListingsContainer, DynamicListingModel, selected_model)
        
        # Save formatted data
        save_formatted_data(formatted_data, output_folder, f'sorted_data_{file_number}.json', f'sorted_data_{file_number}.xlsx')

        # Return token counts and formatted data
        input_tokens = token_counts.get('input_tokens', 0)
        output_tokens = token_counts.get('output_tokens', 0)
        return input_tokens, output_tokens, formatted_data

    except Exception as e:
        return 0, 0, None
        
