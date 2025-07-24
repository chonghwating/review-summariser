"""
Fast-loading Streamlit UI for the Cafe & Restaurant Analyzer
Optimized for quick startup and minimal dependencies
"""

import streamlit as st
import os
import json
import time
from datetime import datetime

# Only import what we absolutely need for startup
try:
    from cafe_analyzer_core import (
        EnhancedCafeAnalyzer, 
        LocationData, 
        BusinessInfo, 
        ReviewData
    )
    CORE_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Core module import failed: {e}")
    CORE_AVAILABLE = False

# Enhanced page configuration
st.set_page_config(
    page_title="🎯 Smart Cafe & Restaurant Analyzer",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_custom_css():
    """Apply custom CSS styling"""
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-success { color: #28a745; font-weight: bold; }
    .status-warning { color: #ffc107; font-weight: bold; }
    .status-error { color: #dc3545; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    """Render the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Smart Cafe & Restaurant Analyzer</h1>
        <p>AI-powered analysis with advanced insights and work-friendliness detection</p>
    </div>
    """, unsafe_allow_html=True)

def check_system_status():
    """Check system status and display diagnostics"""
    st.header("🔧 System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Check core module
        if CORE_AVAILABLE:
            st.success("✅ Core Module Loaded")
        else:
            st.error("❌ Core Module Failed")
    
    with col2:
        # Check API key
        api_key = os.getenv("SERPAPI", "")
        if api_key:
            st.success("✅ API Key Found")
        else:
            st.error("❌ API Key Missing")
    
    with col3:
        # Check dependencies
        try:
            import requests
            st.success("✅ Dependencies OK")
        except ImportError:
            st.error("❌ Missing Dependencies")
    
    return CORE_AVAILABLE and bool(api_key)

def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    
    if 'selected_business' not in st.session_state:
        st.session_state.selected_business = None
    
    if 'reviews' not in st.session_state:
        st.session_state.reviews = None
    
    if 'analysis' not in st.session_state:
        st.session_state.analysis = None

def render_quick_search():
    """Render a simplified search interface"""
    st.header("🔍 Quick Search")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Enter cafe or restaurant name:",
            placeholder="e.g., Starbucks, Blue Bottle Coffee, Joe's Pizza",
            help="Include location for better results"
        )
    
    with col2:
        search_button = st.button("🚀 Search", type="primary", use_container_width=True)
    
    if search_button and search_query:
        if not st.session_state.analyzer:
            with st.spinner("🔧 Initializing analyzer..."):
                try:
                    api_key = os.getenv("SERPAPI", "")
                    st.session_state.analyzer = EnhancedCafeAnalyzer(api_key)
                    st.success("✅ Analyzer ready!")
                except Exception as e:
                    st.error(f"❌ Failed to initialize: {e}")
                    return
        
        # Perform search
        with st.spinner(f"🔍 Searching for: {search_query}"):
            try:
                results = st.session_state.analyzer.search_places(search_query, max_results=10)
                
                if "error" in results:
                    st.error(f"❌ Search failed: {results['error']}")
                elif not results.get("places"):
                    st.error(f"❌ No results found for: {search_query}")
                else:
                    st.session_state.search_results = results
                    st.success(f"✅ Found {len(results['places'])} businesses")
                    render_search_results()
                    
            except Exception as e:
                st.error(f"❌ Search error: {e}")

def render_search_results():
    """Display search results"""
    if not st.session_state.search_results:
        return
    
    st.subheader("🏪 Search Results")
    
    places = st.session_state.search_results["places"]
    
    for i, business in enumerate(places):
        with st.expander(f"☕ {business.title} - ⭐ {business.rating}/5 ({business.reviews_count} reviews)"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**📍 Address:** {business.address}")
                st.markdown(f"**📞 Phone:** {business.phone or 'Not available'}")
                if business.website:
                    st.markdown(f"**🌐 Website:** [Visit]({business.website})")
                st.markdown(f"**🏷️ Type:** {business.business_type}")
            
            with col2:
                if st.button(f"📊 Analyze Reviews", key=f"analyze_{i}"):
                    st.session_state.selected_business = business
                    analyze_business(business)

def analyze_business(business):
    """Analyze a selected business"""
    st.subheader(f"🔍 Analyzing: {business.title}")
    
    # Create tabs for results
    tabs = st.tabs(["📊 Quick Analysis", "📝 Reviews", "🗺️ Location"])
    
    with tabs[0]:
        with st.spinner("📊 Fetching and analyzing reviews..."):
            try:
                # Fetch reviews
                reviews = st.session_state.analyzer.fetch_reviews(business, max_reviews=20)
                
                if not reviews:
                    st.error("❌ No reviews found")
                    return
                
                # Quick analysis
                avg_rating = sum(r.rating for r in reviews) / len(reviews)
                review_texts = " ".join([r.text for r in reviews if r.text])
                
                # Display quick metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("📊 Reviews Analyzed", len(reviews))
                
                with col2:
                    st.metric("⭐ Average Rating", f"{avg_rating:.1f}/5")
                
                with col3:
                    # Simple work-friendly check
                    work_indicators = ['wifi', 'laptop', 'work', 'study', 'quiet']
                    work_mentions = sum(1 for word in work_indicators if word in review_texts.lower())
                    st.metric("💻 Work Mentions", work_mentions)
                
                # Quick insights
                st.subheader("💡 Quick Insights")
                
                if avg_rating >= 4.0:
                    st.success("🟢 Highly rated establishment")
                elif avg_rating >= 3.0:
                    st.info("🟡 Average rated establishment")
                else:
                    st.warning("🟠 Below average ratings")
                
                if work_mentions >= 3:
                    st.success("💻 Appears work-friendly based on reviews")
                elif work_mentions >= 1:
                    st.info("💻 Some work-friendly mentions")
                else:
                    st.info("💻 Limited work-friendly indicators")
                
            except Exception as e:
                st.error(f"❌ Analysis error: {e}")
    
    with tabs[1]:
        if 'reviews' in locals():
            st.subheader("📝 Recent Reviews")
            for i, review in enumerate(reviews[:10]):
                with st.expander(f"Review {i+1}: {review.author} - {'⭐' * review.rating}"):
                    st.markdown(f"**Date:** {review.date}")
                    st.markdown(f"**Rating:** {review.rating}/5")
                    st.markdown(f"**Review:** {review.text}")
        else:
            st.info("Run analysis first to see reviews")
    
    with tabs[2]:
        st.subheader("🗺️ Location Information")
        st.markdown(f"**Name:** {business.title}")
        st.markdown(f"**Address:** {business.address}")
        if business.coordinates:
            st.markdown(f"**Coordinates:** {business.coordinates.latitude:.6f}, {business.coordinates.longitude:.6f}")
        else:
            st.info("📍 Coordinates not available")

def render_setup_guide():
    """Render setup guide for users"""
    st.header("🚀 Setup Guide")
    
    st.markdown("""
    ### 📋 Prerequisites:
    1. **SerpAPI Key** - Get one from [serpapi.com](https://serpapi.com)
    2. **Python Environment** - Make sure you have the required packages
    
    ### 🔧 Setup Steps:
    
    **1. Set your API key:**
    ```bash
    export SERPAPI=your_serpapi_key_here
    ```
    
    **2. Install required packages:**
    ```bash
    pip install streamlit requests pandas plotly folium streamlit-folium
    ```
    
    **3. Optional AI features:**
    ```bash
    pip install transformers torch
    ```
    
    ### ✅ Verification:
    Use the system status above to verify everything is working correctly.
    """)

def main():
    """Fast-loading main application"""
    try:
        apply_custom_css()
        render_header()
        
        # Quick system check
        system_ok = check_system_status()
        
        if not system_ok:
            render_setup_guide()
            return
        
        initialize_session_state()
        
        # Main interface
        render_quick_search()
        
        # Show selected business analysis
        if st.session_state.selected_business:
            st.markdown("---")
            analyze_business(st.session_state.selected_business)
        
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        
        with st.expander("🔧 Debug Information"):
            st.code(str(e))
            st.markdown("**Possible solutions:**")
            st.markdown("1. Check your internet connection")
            st.markdown("2. Verify your API key is set correctly")
            st.markdown("3. Make sure all required files are present")
            st.markdown("4. Try refreshing the page")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🎯 <strong>Smart Cafe & Restaurant Analyzer</strong> - Fast Loading Version</p>
        <p><small>Optimized for quick startup and essential functionality</small></p>
    </div>
    """, unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    main()
else:
    main()  # For streamlit run