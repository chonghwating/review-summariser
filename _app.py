"""
Streamlit UI for the Cafe & Restaurant Analyzer
Clean separation between UI and business logic
"""

import streamlit as st
import os
import json
import time
from datetime import datetime
import pandas as pd
from streamlit_folium import folium_static

# Import core business logic
from cafe_analyzer_core import (
    EnhancedCafeAnalyzer, 
    LocationData, 
    BusinessInfo, 
    ReviewData,
    VisualizationManager,
    InteractiveMap,
    ReportGenerator
)

# Enhanced page configuration
st.set_page_config(
    page_title="🎯 Smart Cafe & Restaurant Analyzer",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/issues',
        'Report a bug': 'https://github.com/your-repo/issues',
        'About': "Advanced cafe & restaurant analysis tool powered by AI"
    }
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
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
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

def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'analyzer' not in st.session_state:
        api_key = os.getenv("SERPAPI", "")
        if api_key:
            try:
                st.session_state.analyzer = EnhancedCafeAnalyzer(api_key)
            except Exception as e:
                st.session_state.analyzer = None
                st.error(f"Failed to initialize analyzer: {e}")
        else:
            st.session_state.analyzer = None
    
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    
    if 'selected_business' not in st.session_state:
        st.session_state.selected_business = None
    
    if 'reviews' not in st.session_state:
        st.session_state.reviews = None
    
    if 'analysis' not in st.session_state:
        st.session_state.analysis = None
    
    if 'user_location' not in st.session_state:
        st.session_state.user_location = None
    
    if 'show_location_detector' not in st.session_state:
        st.session_state.show_location_detector = False

def render_sidebar():
    """Render the sidebar with settings and configuration"""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Status
        with st.expander("🔑 API Status", expanded=False):
            if st.session_state.analyzer:
                try:
                    is_valid, message = st.session_state.analyzer.validate_api_key()
                    if is_valid:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")
                except Exception as e:
                    st.error(f"❌ API validation failed: {e}")
            else:
                st.error("❌ No API key configured")
        
        # Enhanced settings
        st.subheader("📊 Analysis Settings")
        max_reviews = st.slider("Number of reviews to analyze", 10, 100, 30, 5)
        enable_ai = st.checkbox("Enable AI summarization", value=True)
        show_advanced = st.checkbox("Show advanced metrics", value=False)
        
        # Location settings
        render_location_settings()
        
        return max_reviews, enable_ai, show_advanced

def render_location_settings():
    """Render location configuration section"""
    st.subheader("📍 Location Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🌍 Auto-Detect", help="Use browser geolocation"):
            st.session_state.show_location_detector = True
    
    with col2:
        if st.button("🗑️ Clear", help="Clear current location"):
            st.session_state.user_location = None
            st.session_state.show_location_detector = False
            st.rerun()
    
    # Manual location input
    with st.expander("📍 Manual Location Entry"):
        manual_lat = st.number_input("Latitude", value=0.0, format="%.6f")
        manual_lng = st.number_input("Longitude", value=0.0, format="%.6f")
        
        if st.button("📌 Set Manual Location"):
            if manual_lat != 0.0 and manual_lng != 0.0:
                st.session_state.user_location = LocationData(
                    latitude=manual_lat,
                    longitude=manual_lng,
                    source="manual"
                )
                st.success("📍 Manual location set!")
                st.rerun()
    
    # Quick location presets
    st.subheader("🌍 Quick Locations")
    preset_locations = {
        "🇺🇸 New York": LocationData(40.7128, -74.0060, "preset"),
        "🇮🇩 Jakarta": LocationData(-6.2088, 106.8456, "preset"),
        "🇬🇧 London": LocationData(51.5074, -0.1278, "preset"),
        "🇫🇷 Paris": LocationData(48.8566, 2.3522, "preset"),
        "🇯🇵 Tokyo": LocationData(35.6762, 139.6503, "preset"),
    }
    
    selected_preset = st.selectbox("Select preset location:", 
                                 ["None"] + list(preset_locations.keys()))
    
    if selected_preset != "None" and st.button("🎯 Use Preset"):
        st.session_state.user_location = preset_locations[selected_preset]
        st.success(f"📍 Set to {selected_preset}")
        st.rerun()
    
    # Display current location
    if st.session_state.user_location:
        loc = st.session_state.user_location
        st.success("📍 Location Active")
        st.info(f"📍 {loc.latitude:.4f}, {loc.longitude:.4f}")
        st.caption(f"Source: {loc.source}")

def render_search_interface():
    """Render the main search interface"""
    st.header("🔍 Find & Analyze Cafes/Restaurants")
    
    # Search method selection
    search_method = st.radio(
        "Choose your search method:",
        ["🏪 Business Name", "🗺️ Google Maps Link"],
        horizontal=True
    )
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        if search_method == "🏪 Business Name":
            search_query = st.text_input(
                "Enter cafe or restaurant name:",
                placeholder="e.g., Starbucks, Blue Bottle Coffee, Joe's Pizza",
                help="Include location for better results (e.g., 'Starbucks Times Square')"
            )
            google_url = None
        else:
            google_url = st.text_input(
                "Paste Google Maps URL:",
                placeholder="https://maps.google.com/maps/place/...",
                help="Supports both full URLs and shortened links (goo.gl, maps.app.goo.gl)"
            )
            search_query = None
    
    with col2:
        search_button = st.button("🚀 Search", type="primary", use_container_width=True)
    
    return search_method, search_query, google_url, search_button

def handle_search(search_method, search_query, google_url, analyzer):
    """Handle the search logic"""
    if search_method == "🏪 Business Name" and search_query:
        user_location = st.session_state.user_location
        results = analyzer.search_places(search_query, user_location, max_results=20)
        
        if "error" in results:
            st.error(f"❌ Search failed: {results['error']}")
        elif not results.get("places"):
            st.error(f"❌ No results found for: {search_query}")
        else:
            st.session_state.search_results = results
            st.success(f"✅ Found {len(results['places'])} businesses")
    
    elif search_method == "🗺️ Google Maps Link" and google_url:
        # Resolve URL if needed
        resolved_url = analyzer.resolve_google_maps_url(google_url)
        if not resolved_url:
            st.error("❌ Failed to resolve Google Maps URL")
            return
        
        # Extract business info
        business_info = analyzer.extract_business_info_from_url(resolved_url)
        if not business_info:
            st.error("❌ Could not extract business information from URL")
            return
        
        # Search using extracted info
        search_term = business_info.get('name') or business_info.get('query', '')
        if not search_term:
            st.error("❌ Could not determine search term from URL")
            return
        
        st.info(f"🔍 Searching for: **{search_term}**")
        
        user_location = st.session_state.user_location
        if not user_location and business_info.get('coordinates'):
            user_location = business_info['coordinates']
        
        results = analyzer.search_places(search_term, user_location)
        
        if "error" in results:
            st.error(f"❌ Search failed: {results['error']}")
        elif not results.get("places"):
            st.error(f"❌ No results found for extracted business: {search_term}")
        else:
            st.session_state.search_results = results
            st.success(f"✅ Found {len(results['places'])} matching businesses")
    else:
        st.error("❌ Please provide either a business name or Google Maps URL")

def render_business_selection():
    """Render business selection interface"""
    if not st.session_state.search_results:
        return None
    
    st.markdown("---")
    st.header("🏪 Select Your Business")
    
    places = st.session_state.search_results["places"]
    
    if len(places) == 1:
        # Auto-select single result
        st.session_state.selected_business = places[0]
        business = places[0]
        st.success(f"✅ Auto-selected: **{business.title}**")
        return business
    else:
        # Multiple results - show selection
        st.subheader("Choose from the following options:")
        
        # Create enhanced dropdown options
        options = []
        for i, business in enumerate(places):
            rating_stars = '⭐' * min(int(business.rating), 5) if business.rating > 0 else '(No rating)'
            distance_text = f" • 📍 {business.distance_km}km" if business.distance_km else ""
            price_text = f" • 💰 {business.price_level}" if business.price_level else ""
            
            option = f"☕ {business.title} • {rating_stars} {business.rating}/5 • ({business.reviews_count} reviews){distance_text}{price_text}"
            options.append(option)
        
        selected_index = st.selectbox(
            "Select your business:",
            range(len(options)),
            format_func=lambda x: options[x],
            key="business_selector"
        )
        
        st.session_state.selected_business = places[selected_index]
        business = places[selected_index]
        
        # Show detailed business info
        with st.expander("📋 Business Details", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📍 Basic Information:**")
                st.markdown(f"**Name:** {business.title}")
                st.markdown(f"**Rating:** ⭐ {business.rating}/5 ({business.reviews_count} reviews)")
                st.markdown(f"**Address:** {business.address}")
                st.markdown(f"**Type:** {business.business_type}")
                
            with col2:
                st.markdown("**📞 Contact & Details:**")
                st.markdown(f"**Phone:** {business.phone or 'Not available'}")
                if business.website:
                    st.markdown(f"**Website:** [Visit Website]({business.website})")
                if business.hours:
                    st.markdown(f"**Hours:** {business.hours}")
                if business.price_level:
                    st.markdown(f"**Price Level:** {business.price_level}")
        
        return business

def render_analysis_interface(business, analyzer, max_reviews, enable_ai):
    """Render the analysis interface with enhanced progress tracking"""
    if not business:
        return
    
    st.markdown("---")
    st.header("🔍 Analyze Reviews")
    
    # Show current analysis status
    if st.session_state.reviews and st.session_state.analysis:
        st.success(f"✅ Analysis complete! Found {len(st.session_state.reviews)} reviews")
        
        # Show quick stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_rating = st.session_state.analysis.get("rating_distribution", {}).get("average", 0)
            st.metric("⭐ Avg Rating", f"{avg_rating:.1f}/5")
        with col2:
            sentiment = st.session_state.analysis.get("sentiment_analysis", {}).get("sentiment_label", "N/A")
            st.metric("😊 Sentiment", sentiment)
        with col3:
            work_score = st.session_state.analysis.get("work_friendliness", {}).get("total_score", 0)
            st.metric("💻 Work Score", f"{work_score}/10")
        with col4:
            total_reviews = len(st.session_state.reviews)
            st.metric("📝 Reviews", total_reviews)
        
        return
    
    # Analysis button and progress interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        start_analysis = st.button("🚀 Start Analysis", type="primary", use_container_width=True)
    
    with col2:
        if st.button("🔄 Clear Cache", help="Clear cached data and start fresh"):
            # Clear cache and session state
            if hasattr(st.session_state, 'analyzer') and st.session_state.analyzer:
                st.session_state.analyzer.cache = st.session_state.analyzer.cache.__class__()
            for key in ['reviews', 'analysis']:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("🗑️ Cache cleared!")
            st.rerun()
    
    if start_analysis:
        # Create progress containers
        progress_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Fetch reviews with progress tracking
            def progress_callback(progress, message):
                progress_bar.progress(progress * 0.6)  # Reviews fetch is 60% of total
                status_text.text(message)
            
            st.info(f"📝 Fetching up to {max_reviews} reviews for **{business.title}**")
            
            try:
                reviews = analyzer.fetch_reviews(business, max_reviews, progress_callback)
                
                if not reviews:
                    st.error("❌ No reviews found for this business")
                    return
                
                st.session_state.reviews = reviews
                progress_bar.progress(0.6)
                status_text.text(f"✅ Successfully fetched {len(reviews)} reviews")
                
            except Exception as e:
                st.error(f"❌ Error fetching reviews: {str(e)}")
                return
            
            # Step 2: Analyze reviews
            progress_bar.progress(0.7)
            status_text.text("🔬 Running sentiment analysis and extracting insights...")
            
            try:
                analysis = analyzer.analyze_reviews(reviews)
                
                if "error" in analysis:
                    st.error(f"❌ Analysis failed: {analysis['error']}")
                    return
                
                st.session_state.analysis = analysis
                progress_bar.progress(0.9)
                status_text.text("✅ Review analysis completed")
                
            except Exception as e:
                st.error(f"❌ Error analyzing reviews: {str(e)}")
                return
            
            # Step 3: Generate AI summary (if enabled)
            if enable_ai:
                status_text.text("🤖 Generating AI summary...")
                
                try:
                    ai_summary = analyzer.generate_ai_summary(reviews, analysis)
                    progress_bar.progress(1.0)
                    status_text.text("✅ AI summary generated!")
                    
                    # Show preview of AI summary
                    with st.expander("🤖 AI Summary Preview", expanded=True):
                        st.markdown(ai_summary)
                        
                except Exception as e:
                    st.warning(f"⚠️ AI summary failed: {str(e)}")
                    progress_bar.progress(1.0)
                    status_text.text("📊 Manual analysis completed successfully")
            else:
                progress_bar.progress(1.0)
                status_text.text("✅ Analysis completed (AI summarization disabled)")
            
            # Final success message
            time.sleep(0.5)  # Brief pause for UX
            st.balloons()
            st.success("🎉 Complete analysis ready! Check the tabs below for detailed insights.")
            
            # Show quick preview stats
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_rating = analysis.get("rating_distribution", {}).get("average", 0)
                st.info(f"⭐ **Average Rating:** {avg_rating:.1f}/5")
            with col2:
                work_score = analysis.get("work_friendliness", {}).get("total_score", 0)
                st.info(f"💻 **Work-Friendly:** {work_score}/10")
            with col3:
                sentiment = analysis.get("sentiment_analysis", {}).get("sentiment_label", "Neutral")
                st.info(f"😊 **Sentiment:** {sentiment}")
            
            # Auto-scroll hint
            st.markdown("⬇️ **Scroll down to explore detailed analysis tabs**")

def render_results_tabs(business, reviews, analysis, analyzer, enable_ai):
    """Render the tabbed results interface"""
    if not reviews or not analysis:
        return
    
    st.markdown("---")
    st.header("📊 Analysis Results")
    
    # Create tabs for different views
    tabs = st.tabs([
        "📊 Overview", 
        "📈 Visualizations", 
        "🗺️ Location & Map", 
        "💻 Work Analysis", 
        "📝 Review Explorer"
    ])
    
    with tabs[0]:  # Overview
        render_overview_tab(business, reviews, analysis, analyzer, enable_ai)
    
    with tabs[1]:  # Visualizations
        render_visualizations_tab(analysis)
    
    with tabs[2]:  # Location & Map
        render_location_tab(business)
    
    with tabs[3]:  # Work Analysis
        render_work_analysis_tab(analysis)
    
    with tabs[4]:  # Review Explorer
        render_review_explorer_tab(reviews, analyzer)

def render_overview_tab(business, reviews, analysis, analyzer, enable_ai):
    """Render the overview tab"""
    st.subheader("📊 Analysis Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    rating_avg = analysis.get("rating_distribution", {}).get("average", 0)
    work_score = analysis.get("work_friendliness", {}).get("total_score", 0)
    sentiment_label = analysis.get("sentiment_analysis", {}).get("sentiment_label", "Neutral")
    
    with col1:
        st.metric("📊 Total Reviews", len(reviews))
    
    with col2:
        st.metric("⭐ Average Rating", f"{rating_avg:.1f}/5")
    
    with col3:
        st.metric("💻 Work-Friendly Score", f"{work_score}/10")
    
    with col4:
        st.metric("😊 Overall Sentiment", sentiment_label)
    
    # Business summary
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🏪 Business Summary")
        st.markdown(f"**{business.title}**")
        st.markdown(f"📍 {business.address}")
        st.markdown(f"⭐ {business.rating}/5 stars ({business.reviews_count} total reviews)")
        if business.phone:
            st.markdown(f"📞 {business.phone}")
        if business.website:
            st.markdown(f"🌐 [Visit Website]({business.website})")
    
    with col2:
        work_data = analysis.get("work_friendliness", {})
        status = work_data.get("status", "Not assessed")
        
        if "🟢" in status:
            st.success(status)
        elif "🟡" in status:
            st.warning(status)
        elif "🟠" in status:
            st.info(status)
        else:
            st.error(status)
    
    # Quick insights
    st.markdown("---")
    st.subheader("🎯 Quick Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🍽️ Popular Menu Items:**")
        menu_items = analysis.get("menu_items", [])[:5]
        if menu_items:
            for item in menu_items:
                st.markdown(f"• {item}")
        else:
            st.markdown("• No specific items frequently mentioned")
    
    with col2:
        st.markdown("**📈 Key Themes:**")
        themes = analysis.get("key_themes", {})
        sorted_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)[:5]
        for theme, count in sorted_themes:
            st.markdown(f"• {theme.replace('_', ' ').title()}: {count} mentions")

def render_visualizations_tab(analysis):
    """Render the visualizations tab"""
    st.subheader("📈 Data Visualizations")
    
    # Rating distribution chart
    rating_fig = VisualizationManager.create_rating_distribution_chart(analysis)
    if rating_fig:
        st.plotly_chart(rating_fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment radar chart
        sentiment_fig = VisualizationManager.create_sentiment_radar_chart(analysis)
        if sentiment_fig:
            st.plotly_chart(sentiment_fig, use_container_width=True)
    
    with col2:
        # Work-friendly gauge
        work_fig = VisualizationManager.create_work_friendly_gauge(analysis)
        if work_fig:
            st.plotly_chart(work_fig, use_container_width=True)
    
    # Menu popularity chart
    menu_items = analysis.get("menu_items", [])
    menu_fig = VisualizationManager.create_menu_popularity_chart(menu_items)
    if menu_fig:
        st.plotly_chart(menu_fig, use_container_width=True)
    else:
        st.info("📊 No menu items were frequently mentioned in reviews")

def render_location_tab(business):
    """Render the location and map tab"""
    st.subheader("🗺️ Location & Interactive Map")
    
    # Business location details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📍 Business Information:**")
        st.markdown(f"**Name:** {business.title}")
        st.markdown(f"**Address:** {business.address}")
        st.markdown(f"**Phone:** {business.phone or 'Not available'}")
        st.markdown(f"**Type:** {business.business_type}")
        if business.hours:
            st.markdown(f"**Hours:** {business.hours}")
    
    with col2:
        st.markdown("**📊 Location Metrics:**")
        if business.coordinates:
            st.markdown(f"**Coordinates:** {business.coordinates.latitude:.6f}, {business.coordinates.longitude:.6f}")
        if business.distance_km:
            st.markdown(f"**Distance from you:** {business.distance_km} km")
        if business.website:
            st.markdown(f"**Website:** [Visit Website]({business.website})")
        if business.price_level:
            st.markdown(f"**Price Level:** {business.price_level}")
    
    # Interactive map
    if business.coordinates:
        st.markdown("**🗺️ Interactive Map:**")
        user_location = st.session_state.user_location
        business_map = InteractiveMap.create_business_map(business, user_location)
        
        if business_map:
            folium_static(business_map, width=700, height=500)
        else:
            st.warning("📍 Could not generate map - coordinates unavailable")
    else:
        st.warning("📍 Location coordinates not available for mapping")

def render_work_analysis_tab(analysis):
    """Render the work-friendliness analysis tab"""
    st.subheader("💻 Work-Friendliness Detailed Analysis")
    
    work_data = analysis.get("work_friendliness", {})
    
    # Overall status
    st.markdown(f"### {work_data.get('status', 'Status Unknown')}")
    
    # Score breakdown
    scores = work_data.get("scores", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Feature Scores:**")
        for feature, score in scores.items():
            feature_name = feature.replace('_', ' ').title()
            bar_length = min(score * 2, 10)  # Scale for visual
            bar = "█" * bar_length + "░" * (10 - bar_length)
            st.markdown(f"**{feature_name}:** {score} {bar}")
    
    with col2:
        st.markdown("**💡 Work-Friendly Insights:**")
        
        total_score = work_data.get("total_score", 0)
        
        if total_score >= 5:
            st.success("🟢 Excellent for remote work and studying")
            st.markdown("• Strong WiFi and power availability")
            st.markdown("• Conducive atmosphere for focus")
            st.markdown("• Frequently mentioned by remote workers")
        elif total_score >= 3:
            st.warning("🟡 Decent for light work sessions")
            st.markdown("• Some work-friendly features available")
            st.markdown("• May be suitable for casual work")
            st.markdown("• Check specific times for crowd levels")
        elif total_score >= 1:
            st.info("🟠 Limited work features")
            st.markdown("• Basic amenities may be available")
            st.markdown("• Better suited for social meetings")
            st.markdown("• Consider for short work sessions only")
        else:
            st.error("🔴 Not recommended for work")
            st.markdown("• Lacks essential work amenities")
            st.markdown("• Atmosphere not conducive to focus")
            st.markdown("• Better for leisure dining only")

def render_review_explorer_tab(reviews, analyzer):
    """Render the review explorer tab"""
    st.subheader("📝 Review Explorer")
    
    # Filter controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rating_filter = st.multiselect(
            "Filter by rating:",
            [1, 2, 3, 4, 5],
            default=[1, 2, 3, 4, 5],
            key="rating_filter"
        )
    
    with col2:
        min_length = st.slider(
            "Minimum review length:",
            0, 500, 0,
            help="Filter out very short reviews"
        )
    
    with col3:
        sort_option = st.selectbox(
            "Sort reviews by:",
            ["Rating (High to Low)", "Rating (Low to High)", "Most Liked", "Longest First", "Most Recent"]
        )
    
    # Apply filters
    filtered_reviews = [
        r for r in reviews 
        if r.rating in rating_filter and len(r.text) >= min_length
    ]
    
    # Sort reviews
    if sort_option == "Rating (High to Low)":
        filtered_reviews.sort(key=lambda x: x.rating, reverse=True)
    elif sort_option == "Rating (Low to High)":
        filtered_reviews.sort(key=lambda x: x.rating)
    elif sort_option == "Most Liked":
        filtered_reviews.sort(key=lambda x: x.likes, reverse=True)
    elif sort_option == "Longest First":
        filtered_reviews.sort(key=lambda x: len(x.text), reverse=True)
    # "Most Recent" is default order
    
    st.info(f"📊 Showing {len(filtered_reviews)} of {len(reviews)} reviews")
    
    # Display reviews
    for i, review in enumerate(filtered_reviews[:20], 1):  # Limit to 20 for performance
        with st.expander(f"Review {i}: {review.author} ({'⭐' * review.rating} {review.rating}/5)"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**📅 Date:** {review.date}")
                st.markdown(f"**⭐ Rating:** {'⭐' * review.rating} ({review.rating}/5)")
                st.markdown(f"**📝 Review:**")
                st.markdown(review.text)
                
                # Show business response if available
                if review.business_response:
                    st.markdown("**🏪 Business Response:**")
                    response_text = review.business_response.get("text", "")
                    if response_text:
                        st.markdown(f"*{response_text}*")
            
            with col2:
                if review.likes > 0:
                    st.metric("👍 Likes", review.likes)
                
                if review.author_reviews_count > 0:
                    st.metric("👤 Author's Reviews", review.author_reviews_count)
                
                # Sentiment analysis for individual review
                try:
                    review_sentiment = analyzer.sentiment_analyzer.analyze_sentiment(review.text)
                    sentiment_score = review_sentiment.get("sentiment_score", 0)
                    
                    if sentiment_score > 0:
                        st.success(f"😊 Positive ({sentiment_score})")
                    elif sentiment_score < 0:
                        st.error(f"😞 Negative ({sentiment_score})")
                    else:
                        st.info("😐 Neutral")
                except:
                    st.info("😐 Sentiment N/A")

def render_footer():
    """Render the application footer"""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🎯 <strong>Smart Cafe & Restaurant Analyzer v2.0</strong></p>
        <p>Powered by AI • Enhanced Analytics • Real-time Data</p>
        <p><small>Data sourced from Google Reviews via SerpAPI</small></p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Enhanced main application with better error handling"""
    try:
        apply_custom_css()
        render_header()
        
        # Check for API key first
        api_key = os.getenv("SERPAPI", "")
        if not api_key:
            st.error("🔑 **Configuration Error**: SerpAPI key not found in environment variables.")
            st.markdown("""
            **Setup Instructions:**
            1. Get your API key from [SerpAPI](https://serpapi.com)
            2. Set environment variable: `export SERPAPI=your_key_here`
            3. Restart the application
            """)
            st.stop()
        
        initialize_session_state()
        
        if not st.session_state.analyzer:
            st.error("❌ Failed to initialize analyzer. Please check your API key and try again.")
            if st.button("🔄 Retry Initialization"):
                try:
                    st.session_state.analyzer = EnhancedCafeAnalyzer(api_key)
                    st.success("✅ Analyzer initialized successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Initialization failed: {e}")
            st.stop()
        
        analyzer = st.session_state.analyzer
        
        # Render sidebar and get settings
        max_reviews, enable_ai, show_advanced = render_sidebar()
        
        # Render main search interface
        search_method, search_query, google_url, search_button = render_search_interface()
        
        # Handle search
        if search_button:
            try:
                handle_search(search_method, search_query, google_url, analyzer)
            except Exception as e:
                st.error(f"❌ Search error: {str(e)}")
                st.info("Please try again or contact support if the issue persists.")
        
        # Render business selection
        try:
            selected_business = render_business_selection()
        except Exception as e:
            st.error(f"❌ Error in business selection: {str(e)}")
            selected_business = None
        
        # Render analysis interface
        if selected_business:
            try:
                render_analysis_interface(selected_business, analyzer, max_reviews, enable_ai)
                
                # Render results if available
                if st.session_state.reviews and st.session_state.analysis:
                    render_results_tabs(
                        selected_business, 
                        st.session_state.reviews, 
                        st.session_state.analysis, 
                        analyzer, 
                        enable_ai
                    )
            except Exception as e:
                st.error(f"❌ Error in analysis: {str(e)}")
                st.info("Please try with a different business or contact support.")
        
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        st.markdown("**Troubleshooting:**")
        st.markdown("1. Refresh the page")
        st.markdown("2. Check your internet connection")
        st.markdown("3. Verify your API key is valid")
        
        if st.button("🔄 Restart Application"):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    finally:
        # Always render footer
        render_footer()

# Run the main application directly (Streamlit style)
main()