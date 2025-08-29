# streamlit_app.py (replace your current file with this)
from annotated_text import annotated_text
from bs4 import BeautifulSoup
from gramformer import Gramformer
import streamlit as st
import pandas as pd
import torch
import math
import re
import json
import requests
from streamlit_lottie import st_lottie
from pathlib import Path
import streamlit.components.v1 as components

# --------- utils ----------
def set_seed(seed=1212):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(1212)

@st.cache_resource(show_spinner=False)
def load_gf(model_key):
    return Gramformer(models=model_key, use_gpu=False)

def load_lottie_url(url):
    try:
        r = requests.get(url)
        return r.json()
    except:
        return None

def local_css(path: str):
    css_file = Path(path)
    if css_file.exists():
        st.markdown(f"<style>{css_file.read_text()}</style>", unsafe_allow_html=True)
    else:
        # Fallback clean CSS if file doesn't exist
        st.markdown("""
        <style>
        /* Clean UI Styles */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        
        .header h1 {
            font-size: 3rem !important;
            font-weight: 700 !important;
            margin: 0 0 0.5rem 0 !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .header p {
            font-size: 1.1rem !important;
            line-height: 1.6 !important;
            margin-bottom: 2rem !important;
            opacity: 0.8;
        }
        
        .result-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .result-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
            border-color: rgba(255, 255, 255, 0.2);
        }
        
        .input-section {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1.5rem 0;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .action-buttons {
            display: flex;
            gap: 1rem;
            align-items: center;
            margin: 1.5rem 0;
            flex-wrap: wrap;
        }
        
        .stButton > button {
            border-radius: 8px !important;
            font-weight: 600 !important;
            padding: 0.5rem 1.5rem !important;
            border: none !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton > button:first-child {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
        }
        
        .stButton > button:first-child:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
        }
        
        .copy-btn {
            background: rgba(255, 255, 255, 0.1) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            color: inherit !important;
            border-radius: 6px !important;
            padding: 0.4rem 1rem !important;
            font-size: 0.85rem !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
        }
        
        .copy-btn:hover {
            background: rgba(255, 255, 255, 0.2) !important;
            transform: translateY(-1px) !important;
        }
        
        .metric-card {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        
        .section-header {
            font-size: 1.4rem !important;
            font-weight: 600 !important;
            margin: 2rem 0 1rem 0 !important;
            padding-bottom: 0.5rem !important;
            border-bottom: 2px solid rgba(255, 255, 255, 0.1) !important;
        }
        
        .sidebar .stSlider > div > div > div > div {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        }
        
        .tip-box {
            background: rgba(102, 126, 234, 0.1);
            border-left: 4px solid #667eea;
            border-radius: 0 8px 8px 0;
            padding: 1rem 1.25rem;
            margin: 1rem 0;
        }
        
        .stTextArea textarea {
            border-radius: 8px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            background: rgba(255, 255, 255, 0.02) !important;
        }
        
        .stTextArea textarea:focus {
            border-color: #667eea !important;
            box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
        }
        
        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)

# small JS copy button (uses streamlit components)
def copy_button(text: str, key: str):
    safe_text = json.dumps(text)
    html = f"""
    <button id="btn_{key}" class="copy-btn">📋 Copy</button>
    <script>
    const btn = document.getElementById("btn_{key}");
    btn.onclick = () => {{
        navigator.clipboard.writeText({safe_text});
        const original = btn.innerHTML;
        btn.innerHTML = "✅ Copied!";
        btn.style.background = "rgba(76, 175, 80, 0.2)";
        btn.style.borderColor = "rgba(76, 175, 80, 0.4)";
        setTimeout(() => {{
            btn.innerHTML = original;
            btn.style.background = "rgba(255, 255, 255, 0.1)";
            btn.style.borderColor = "rgba(255, 255, 255, 0.2)";
        }}, 1500);
    }};
    </script>
    """
    components.html(html, height=45)

# ---------- app ----------
def show_edits_table(gf, original, corrected):
    try:
        edits = gf.get_edits(original, corrected)
        if not edits:
            st.info("No edits detected - text appears to be already correct!")
            return
        
        df = pd.DataFrame(edits, columns=['type','original word','orig start','orig end','correct word','corr start','corr end'])
        df = df.set_index('type')
        
        # Clean table styling
        st.markdown("**Edit Details:**")
        st.dataframe(
            df, 
            use_container_width=True,
            height=min(200, len(df) * 35 + 50)
        )
    except Exception as e:
        st.error(f"Failed to compute edits: {str(e)}")

def main():
    # theme + CSS
    st.set_page_config(
        page_title="Gramformer — AI Grammar Correction", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    local_css("assets/style.css")

    # HERO SECTION
    st.markdown("""
    <div class="header">
        <h1>Gramformer</h1>
        <p>Advanced AI-powered grammar correction tool. Type or paste your text below, customize the settings, and get instant, professional corrections.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)

    # SIDEBAR CONTROLS
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")
        
        model_map = {'Corrector': 1}
        max_candidates = st.slider(
            "🔢 Max corrections to show", 
            min_value=1, 
            max_value=6, 
            value=3,
            help="Number of correction alternatives to generate"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            show_scores = st.checkbox("📊 Show scores", value=True)
        with col2:
            auto_copy = st.checkbox("📋 Copy buttons", value=True)
        
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        - **Longer texts** work better
        - **Check multiple corrections** for best results  
        - **Use the copy button** for quick access
        - **Download CSV** for batch processing
        """)

    # INPUT SECTION

    st.markdown("### 📝 Enter Your Text")
    
    # Initialize session state for input
    if "main_input" not in st.session_state:
        st.session_state["main_input"] = ""
    
    input_text = st.text_area(
        "Text to correct:",
        value=st.session_state.get("main_input", ""),
        height=140,
        placeholder="Type or paste your text here... (e.g., 'I are going to the store tomorrow and buy some groceries.')",
        key="text_input"
    )
    
    # Update session state
    if input_text != st.session_state.get("main_input", ""):
        st.session_state["main_input"] = input_text
    
    st.markdown('</div>', unsafe_allow_html=True)

    # ACTION BUTTONS
    st.markdown('<div class="action-buttons">', unsafe_allow_html=True)
    
    col_a, col_b, col_spacer, col_stats = st.columns([1.2, 1, 2, 1.5])
    
    with col_a:
        correct_clicked = st.button("✨ Correct Grammar", type="primary")
    
    with col_b:
        if st.button("🗑️ Clear Text"):
            st.session_state["main_input"] = ""
            st.rerun()
    
    with col_stats:
        if input_text:
            word_count = len(input_text.split())
            char_count = len(input_text)
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.9rem; opacity: 0.8;">Words: {word_count} | Chars: {char_count}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Add helpful tip
    if not input_text:
        st.markdown("""
        <div class="tip-box">
            <strong>💡 Pro Tip:</strong> Try pasting a paragraph with some grammar mistakes to see Gramformer in action!
        </div>
        """, unsafe_allow_html=True)

    # PROCESSING & RESULTS
    if input_text and correct_clicked:
        with st.spinner("🔄 Processing with Gramformer AI... Please wait"):
            try:
                gf = load_gf(model_map['Corrector'])
                results = gf.correct(input_text, max_candidates=max_candidates)
                if isinstance(results, set):
                    results = list(results)
                
                if not results:
                    st.warning("⚠️ No corrections found. Your text might already be grammatically correct!")
                    return
                
            except Exception as e:
                st.error(f"❌ Error processing text: {str(e)}")
                return

        # RESULTS DISPLAY
        st.markdown(f'<h3 class="section-header">🎯 Grammar Corrections ({len(results)} found)</h3>', unsafe_allow_html=True)
        
        # Prepare data for download
        rows = []
        
        # Results grid
        st.markdown('<div class="results-grid">', unsafe_allow_html=True)
        
        for i, res in enumerate(results):
            if isinstance(res, tuple):
                corrected, score = res
            else:
                corrected, score = res, None
            
            rows.append({
                "original": input_text,
                "correction": corrected, 
                "score": score,
                "rank": i + 1
            })

            # Enhanced result card
            score_display = f"Score: {score:.4f}" if score and show_scores else ""
            confidence = ""
            if score:
                if score > 0.8:
                    confidence = "🟢 High Confidence"
                elif score > 0.6:
                    confidence = "🟡 Medium Confidence" 
                else:
                    confidence = "🔴 Low Confidence"
            
            card_html = f"""
            <div class="result-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                    <div style="font-weight: 600; font-size: 1rem;">Correction #{i+1}</div>
                    <div style="font-size: 0.85rem; opacity: 0.8;">{confidence}</div>
                </div>
                <div style="font-size: 1.1rem; line-height: 1.5; margin-bottom: 1rem; padding: 0.75rem; background: rgba(255, 255, 255, 0.03); border-radius: 6px;">
                    {corrected}
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-size: 0.8rem; opacity: 0.7;">{score_display}</div>
                </div>
            </div>
            """
            
            st.markdown(card_html, unsafe_allow_html=True)
            
            # Copy button
            if auto_copy:
                copy_button(corrected, key=f"correction_{i}")
            
            # Edit details in expander
            with st.expander(f"📋 Show detailed edits for correction #{i+1}", expanded=False):
                show_edits_table(gf, input_text, corrected)
        
        st.markdown('</div>', unsafe_allow_html=True)

        # DOWNLOAD SECTION
        st.markdown("---")
        df_out = pd.DataFrame(rows)
        csv = df_out.to_csv(index=False).encode('utf-8')
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.download_button(
                "📥 Download Results as CSV",
                csv,
                file_name=f"gramformer_corrections_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help="Download all corrections with scores and metadata"
            )

    elif input_text and not correct_clicked:
        st.info("👆 Click the **Correct Grammar** button to generate corrections.")

    # FOOTER
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; opacity: 0.6; font-size: 0.9rem;">
        <p>Powered by Gramformer AI • Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()