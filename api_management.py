import streamlit as st
import os

def get_api_key(api_key_name):
    """
    Get API key from Streamlit session state or environment variables.
    Falls back to environment variables when not in Streamlit context.
    """
    try:
        # Check if we're in a Streamlit context
        if hasattr(st, 'session_state') and st.session_state:
            # Check if the API key from the sidebar is present, else fallback to the .env file
            if api_key_name == 'GOOGLE_API_KEY':
                return st.session_state.get('gemini_api_key') or os.getenv(api_key_name)
            else:
                return os.getenv(api_key_name)
        else:
            # Not in Streamlit context, use environment variables
            return os.getenv(api_key_name)
    except Exception:
        # Fallback to environment variables if any error occurs
        return os.getenv(api_key_name)
