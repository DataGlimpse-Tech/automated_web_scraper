import streamlit as st
import requests
import pandas as pd
from serpapi import GoogleSearch
import google.generativeai as genai
from io import StringIO
import traceback
# Configuration (no fallback values, keeps API keys secure)
SERPAPI_KEY = st.secrets["SERPAPI_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]



# Configure Gemini API safely
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Failed to configure Gemini API: {e}")

#New Chatbot Functions 
def init_conversation():
    """Initialize the conversation history"""
    return [
        {"role": "assistant", "content": "Hello! I'm your data assistant. Please describe the dataset you need."}
    ]

def get_chatbot_response(conversation):
    """Get response from Gemini to refine dataset requirements"""
    prompt = """
    You are a dataset research assistant. Your goal is to understand what kind of dataset the user needs so you can later generate a relevant search query. Ask questions one at a time, in a conversational tone, but make sure each question helps define the type of dataset the user is looking for.

Ask about the following:

What is the main topic or subject the dataset should be about?

Which time period should the data cover? (e.g., latest, 2020–2024, last 10 years)

What geographic area should the data focus on? (e.g., global, India, specific city or region)

Which specific data points or fields are important? (e.g., sales figures, pollution levels, age, sentiment)

Any format or source preferences? (e.g., needs to be in CSV, publicly available, comes from a trusted source)

Ask exactly one question per turn, phrased to help determine what kind of dataset to search for.
Once you have:

A clear topic

A time period

A geographic focus

Key attributes



Respond with:

"READY: [search query]"
    """
    
    # Format conversation history
    history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
    
    full_prompt = f"{prompt}\n\nCurrent conversation:\n{history}"
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(full_prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def extract_search_query(gemini_response):
    """Extract the final search query from Gemini's response"""
    if "READY:" in gemini_response:
        return gemini_response.split("READY:", 1)[1].strip()
    return None

# ---------- Functions ---------- #
def generate_search_query(user_input):
    """Generate a search query that's more likely to find datasets."""
    return f"{user_input} dataset table statistics"

def search_web(query, debug_container=None):
    """Search the web using SerpAPI."""
    params = {
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": 40
    }
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        urls = [
            r['link'] for r in results.get('organic_results', []) 
            if 'link' in r 
            and not r['link'].lower().endswith(('.pdf', '.doc', '.docx'))
        ]
        if debug_container:
            debug_container.write(f"Found {len(urls)} URLs from search results")
        return urls
    except Exception as e:
        error_msg = f"Search API error: {e}"
        st.error(error_msg)
        if debug_container:
            debug_container.write(error_msg)
            debug_container.write(traceback.format_exc())
        return []

def extract_table(url, debug_container=None):
    """Extract tables from a URL."""
    if url.lower().endswith(('.pdf', '.doc', '.docx')):
        if debug_container:
            debug_container.write(f"Skipping PDF/DOC URL: {url}")
        return None, None
    try:
        # Add user agent to avoid blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Log attempt
        if debug_container:
            debug_container.write(f"Attempting to fetch: {url}")
            
        # Try to get the page with timeout
        response = requests.get(url, headers=headers, timeout=10)
        
        # Only proceed if we successfully got content
        if response.status_code == 200:
            try:
                tables = pd.read_html(response.text)
                
                if tables:
                    if debug_container:
                        debug_container.write(f"Found {len(tables)} tables on page")
                        
                    # Find tables with reasonable dimensions
                    valid_tables = [t for t in tables if t.shape[0] >= 3 and t.shape[1] >= 2]
                    
                    if valid_tables:
                        # Find the most relevant table (with most columns)
                        best_table = max(valid_tables, key=lambda t: t.shape[1])
               
                        if debug_container:
                            debug_container.write(f"Selected best table with {best_table.shape[0]} rows and {best_table.shape[1]} columns")
                            
                        return best_table, url
                    elif debug_container:
                        debug_container.write("No valid tables found with sufficient rows/columns")
                elif debug_container:
                    debug_container.write("No tables found on page")
            except Exception as e:
                if debug_container:
                    debug_container.write(f"Error parsing tables: {str(e)}")
                    debug_container.write(traceback.format_exc())
        elif debug_container:
            debug_container.write(f"HTTP status code: {response.status_code}")
            
    except Exception as e:
        if debug_container:
            debug_container.write(f"Error fetching URL: {str(e)}")
            debug_container.write(traceback.format_exc())
            
    return None, None

def clean_table(df, debug_container=None):
    """Clean and normalize a dataframe."""
    if df is None or df.empty:
        if debug_container:
            debug_container.write("Empty or None dataframe received")
        return pd.DataFrame()
        
    try:
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        if debug_container:
            debug_container.write(f"Initial table shape: {df.shape}")
        
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]
    
        # Check if first row might be headers
        unnamed_count = df.columns.to_series().astype(str).str.contains('^unnamed', case=False).sum()
        if unnamed_count > len(df.columns) // 2 and len(df) > 0:
            if debug_container:
                debug_container.write(f"Using first row as header (found {unnamed_count} unnamed columns)")
                
            df.columns = [str(x).strip() for x in df.iloc[0]]
            df = df[1:].reset_index(drop=True)
    
        # Clean column names
        df.columns = df.columns.astype(str)
        df.columns = [col.strip().lower().replace(" ", "_").replace("\n", "_") for col in df.columns]
    
        # Remove unnamed or duplicate columns
        df = df.loc[:, ~pd.Series(df.columns).astype(str).str.contains('^unnamed', case=False).values]
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Drop columns with too many NaN values
        if len(df) > 0:  # Only try this if we have rows
            df = df.dropna(axis=1, thresh=int(0.5 * len(df)))
            
        # Drop rows with all NaN values
        df = df.dropna(how='all')
    
        # Clean data values
        for col in df.columns:
            if df[col].dtype == object:
                # Convert to string and clean common symbols
                df[col] = df[col].astype(str).str.replace(r'[\$\n%°]', '', regex=True).str.strip()
                # Try to convert to numeric if appropriate
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except Exception:
                    pass
    
        df = df.drop_duplicates()
        result = df.reset_index(drop=True)
        
        if debug_container:
            debug_container.write(f"Final cleaned table shape: {result.shape}")
            
        return result
        
    except Exception as e:
        if debug_container:
            debug_container.write(f"Error in clean_table: {str(e)}")
            debug_container.write(traceback.format_exc())
        return pd.DataFrame()

def clean_with_gemini(raw_df, debug_container=None):
    """Clean dataframe using Gemini."""
    if raw_df is None or raw_df.empty:
        if debug_container:
            debug_container.write("Empty dataframe - nothing to clean with Gemini")
        return pd.DataFrame()
    
    # Keep only a reasonable number of rows to send to Gemini
    sample_df = raw_df.head(100)  # Limit to prevent token issues
    
    # Make sure we have valid column names for CSV serialization
    sample_df.columns = [str(col).strip() or f"col_{i}" for i, col in enumerate(sample_df.columns)]
    
    prompt = f"""
You are a data cleaning expert. Clean the following dataset:
1. Fix column names to be descriptive and consistent.
2. Structure the data so it is aligned and readable.
3. Return only the cleaned dataset in raw CSV format. Do not add any explanations or code blocks.
4. Do not reduce the dataset, you can fill the missing values with average values.

Dataset:
{sample_df.to_csv(index=False)}
"""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")  # Updated to a valid model name
        
        if debug_container:
            debug_container.write("Sending data to Gemini for cleaning")
            
        response = model.generate_content(prompt)
        text = response.text.strip()

        if debug_container:
            debug_container.write("Received response from Gemini")
            
        # Extract CSV content from the response
        if "```" in text:
            # Find and extract the CSV block
            parts = text.split("```")
            for part in parts:
                if "," in part and len(part.strip()) > 0:
                    csv_text = part.replace("csv", "").strip()
                    break
            else:
                if debug_container:
                    debug_container.write("No CSV block found in Gemini response")
                raise ValueError("No CSV block found in Gemini response")
        else:
            csv_text = text

        # Remove any markdown formatting that might remain
        csv_text = csv_text.strip()
        
        try:
            cleaned_df = pd.read_csv(StringIO(csv_text))
            
            if debug_container:
                debug_container.write(f"Successfully parsed CSV from Gemini with shape: {cleaned_df.shape}")
                
            # If Gemini returned fewer rows than we have, apply the column names to our original data
            if len(cleaned_df) < len(raw_df) and len(cleaned_df.columns) == len(raw_df.columns):
                if debug_container:
                    debug_container.write("Applying cleaned column names to full dataset")
                result = raw_df.copy()
                result.columns = cleaned_df.columns
                return result
            else:
                return cleaned_df
                
        except Exception as e:
            if debug_container:
                debug_container.write(f"Failed to parse Gemini response as CSV: {e}")
                debug_container.write(traceback.format_exc())
            st.warning(f"Failed to parse Gemini response as CSV: {e}")
            return raw_df
            
    except Exception as e:
        if debug_container:
            debug_container.write(f"Gemini error: {e}")
            debug_container.write(traceback.format_exc())
        st.warning(f"Gemini error: {e}")
        return raw_df  # Return original data if Gemini fails

# ---------- Streamlit UI ---------- #
st.set_page_config(page_title="Auto Dataset Scraper", layout="centered")
st.title("🔍 Auto Dataset Scraper")

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = init_conversation()
if 'search_query' not in st.session_state:
    st.session_state.search_query = None
if 'show_debug' not in st.session_state:
    st.session_state.show_debug = False
if 'dataset_generated' not in st.session_state:
    st.session_state.dataset_generated = False

# Debug toggle in sidebar
with st.sidebar:
    st.session_state.show_debug = st.checkbox("Show debug information", value=st.session_state.show_debug)
    debug_container = st.container() if st.session_state.show_debug else None

    # Add reset button
    if st.button("Reset Conversation"):
        st.session_state.conversation = init_conversation()
        st.session_state.search_query = None
        st.session_state.dataset_generated = False
        st.rerun()

# Display conversation
for message in st.session_state.conversation:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input - only show if dataset hasn't been generated yet
if not st.session_state.dataset_generated:
    if prompt := st.chat_input("Describe your data needs..."):
        # Add user message
        st.session_state.conversation.append({"role": "user", "content": prompt})
        
        # Get assistant response
        with st.spinner("Thinking..."):
            response = get_chatbot_response(st.session_state.conversation)
            st.session_state.conversation.append({"role": "assistant", "content": response})
            
            # Check if we have a final query
            final_query = extract_search_query(response)
            if final_query:
                st.session_state.search_query = final_query
                st.session_state.conversation.append({
                    "role": "assistant",
                    "content": f"Great! I'll search for: **{final_query}**"
                })
        
        st.rerun()

# ========== Dataset Generation ========== #
if st.session_state.search_query and not st.session_state.dataset_generated:
    st.info("Starting dataset generation process...")
    st.session_state.dataset_generated = True
    
    with st.spinner("Searching for relevant data sources..."):
        urls = search_web(st.session_state.search_query, debug_container)
        
        if not urls:
            st.error("No search results found. Please try a different query.")
            st.stop()
        
        st.write(f"Found {len(urls)} potential sources to check")

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_dataframes = []
    total_rows = 0
    sources_checked = 0
    
    for i, url in enumerate(urls):
        progress_percent = int((i / len(urls)) * 100)
        progress_bar.progress(progress_percent)
        
        status_text.text(f"Checking source {i+1}/{len(urls)}: {url[:50]}...")
        df, source_url = extract_table(url, debug_container)
        sources_checked += 1
        
        if df is not None and not df.empty and df.shape[1] > 1:
            cleaned_df = clean_table(df, debug_container)
            if not cleaned_df.empty:
                cleaned_df['source_url'] = source_url
                all_dataframes.append(cleaned_df)
                total_rows += len(cleaned_df)

    progress_bar.progress(100)
    status_text.text(f"Checked {sources_checked} sources. Total rows collected: {total_rows}")
    
    if all_dataframes:
        # Combine all dataframes
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        final_raw_df = combined_df
            
        # Display raw data collected
        with st.expander("Raw Data Collected"):
            st.dataframe(final_raw_df, use_container_width=True)
        
        # Try to clean with Gemini if API key is available
        if GEMINI_API_KEY:
            try:
                with st.spinner("Enhancing dataset with AI..."):
                    final_df = clean_with_gemini(final_raw_df, debug_container)
                st.success("✅ AI-enhanced dataset generated successfully.")
            except Exception as e:
                st.warning(f"⚠️ Gemini cleaning failed: {e}. Using basic cleaning instead.")
                final_df = final_raw_df
        else:
            st.info("Gemini API key not configured. Using basic cleaning only.")
            final_df = final_raw_df
            
        # Display final cleaned data
        st.subheader(f"Final Dataset ({len(final_df)} rows)")
        st.dataframe(final_df, use_container_width=True)
        
        # Download option
        csv = final_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="scraped_dataset.csv",
            mime="text/csv"
        )
    else:
        st.error("❌ Could not find any valid tables from the searched sources. Try rephrasing your query.")
        
   
    
    # Add button to start a new search
    if st.button("Start New Search"):
        st.session_state.conversation = init_conversation()
        st.session_state.search_query = None
        st.session_state.dataset_generated = False
        st.rerun()