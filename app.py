import ssl
# BYPASS SSL VERIFICATION
ssl._create_default_https_context = ssl._create_unverified_context

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
import re

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="VerseVibe", page_icon="✒️")

SHEET_ID = "1GM2BEzSOIJ2l-FHbq6mz-3-WQc7YABEmh1B37T15mHc"
SHEET_GID = "1312416187"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={SHEET_GID}"
MASTER_DOC_URL = "https://docs.google.com/document/d/1GItHNwp82IsjWRiJHPSY2RP9-Jm3l7dZ9Z2Nk9VyGto/edit?tab=t.0"

# --- 2. DATA LOADING ---
@st.cache_data(ttl=600)
def load_data():
    try:
        df = pd.read_csv(CSV_URL)
        if len(df.columns) < 4:
            st.error("Error: The Google Sheet needs at least 4 columns")
            return pd.DataFrame(), []

        # 0:ID, 1:Title, 2:Poet, 3+:Features
        feature_cols = df.columns[3:] 
        df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        return df, feature_cols
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), []

df, feature_cols = load_data()

# --- 3. GOOGLE DOC SCRAPER (PRESERVES FORMATTING) ---
@st.cache_data(show_spinner=False)
def fetch_formatted_poem_from_doc(doc_url, target_title):
    """
    Extracts the HTML and CSS of a specific section in a Google Doc
    to preserve exact formatting.
    """
    try:
        if not isinstance(target_title, str) or "docs.google.com" not in doc_url: return None
        clean_target = target_title.strip().lower()

        # 1. Download HTML Export
        doc_id_match = re.search(r'/d/([a-zA-Z0-9-_]+)', doc_url)
        if not doc_id_match: return None
        doc_id = doc_id_match.group(1)
        
        export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=html"
        response = requests.get(export_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # 2. Extract CSS Styles
        # Google Docs keeps formatting in a <style> block at the top. We need this.
        style_block = soup.find('style')
        css_content = str(style_block) if style_block else ""

        # 3. Find the Heading
        headers = soup.find_all(['h1', 'h2', 'h3']) 
        found_header = None
        for h in headers:
            if clean_target in h.get_text(strip=True).lower():
                found_header = h
                break
        
        # 4. Extract HTML Content
        poem_html_parts = []
        if found_header:
            # Add the CSS first
            poem_html_parts.append(css_content)
            
            # Start a container div
            poem_html_parts.append('<div class="google-doc-content">')
            
            for sibling in found_header.find_next_siblings():
                # Stop at next major header
                if sibling.name in ['h1', 'h2']: 
                    break 
                
                # Keep the HTML tag structure (str(sibling)) instead of just text
                poem_html_parts.append(str(sibling))
            
            poem_html_parts.append('</div>')
            
            # Join everything into one HTML string
            return "".join(poem_html_parts)
        
        return None

    except Exception:
        return None

# --- 4. RECOMMENDATION LOGIC ---
def get_recommendation(current_row, dataframe, features, exclude_ids=[]):
    current_vec = current_row[features].values.reshape(1, -1)
    current_id = current_row.iloc[0] 
    
    candidates = dataframe[
        (dataframe.iloc[:, 0] != current_id) & 
        (~dataframe.iloc[:, 0].isin(exclude_ids))
    ].copy()
    
    if candidates.empty: return None, 0
    
    candidate_matrix = candidates[features].values
    scores = cosine_similarity(current_vec, candidate_matrix)
    best_idx = scores.argmax()
    return candidates.iloc[best_idx], scores[0, best_idx]

# --- 5. UI LAYOUT ---
st.title("VerseVibe ✒️")
st.markdown("Discover poetry based on structural personality.")
st.divider()

if 'current_poem' not in st.session_state: st.session_state.current_poem = None
if 'match_score' not in st.session_state: st.session_state.match_score = None
if 'shown_ids' not in st.session_state: st.session_state.shown_ids = []

col1, col2 = st.columns(2)

with col1:
    if st.button("Show me a poem", use_container_width=True):
        if not df.empty:
            new_poem = df.sample(1).iloc[0]
            st.session_state.current_poem = new_poem
            st.session_state.match_score = None
            st.session_state.shown_ids = [new_poem.iloc[0]] 
        else:
            st.warning("Data empty.")

with col2:
    disabled = st.session_state.current_poem is None
    if st.button("Another poem like this", disabled=disabled, use_container_width=True):
        match, score = get_recommendation(
            st.session_state.current_poem, df, feature_cols,
            exclude_ids=st.session_state.shown_ids
        )
        if match is not None:
            st.session_state.current_poem = match
            st.session_state.match_score = score
            st.session_state.shown_ids.append(match.iloc[0])
        else:
            st.warning("No more matches!")

# --- 6. DISPLAY ---
poem = st.session_state.current_poem

if poem is not None:
    with st.container(border=True):
        title = str(poem.iloc[1]) 
        poet = poem.iloc[2]
        
        st.header(title)
        st.subheader(f"by {poet}")
        
        # SCRAPE FORMATTED CONTENT
        poem_html_content = None
        with st.spinner(f"Reading '{title}'..."):
            found_html = fetch_formatted_poem_from_doc(MASTER_DOC_URL, title)
            if found_html:
                poem_html_content = found_html
            else:
                st.warning(f"Could not find '{title}' in the Master Doc.")

        # RENDER HTML
        if poem_html_content:
            # We use unsafe_allow_html to render the CSS and exact structure
            st.markdown(poem_html_content, unsafe_allow_html=True)
        else:
            st.caption("Text unavailable.")

        # STATS
        if st.session_state.match_score:
            st.divider()
            st.caption(f"Similarity: {int(st.session_state.match_score * 100)}%")
            st.progress(float(st.session_state.match_score))
