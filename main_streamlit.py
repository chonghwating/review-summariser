import streamlit as st
import requests
import json
import time
import re
import urllib.parse
from typing import List, Dict, Optional, Tuple, Union
import pandas as pd
from datetime import datetime, timedelta
import os
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from functools import lru_cache
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
from urllib.parse import urlparse, parse_qs
from collections import Counter, defaultdict
import math
import statistics

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

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('cafe_analyzer.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Data classes for better type safety
@dataclass
class LocationData:
    latitude: float
    longitude: float
    source: str = "unknown"
    accuracy: Optional[float] = None
    timestamp: Optional[datetime] = None

@dataclass
class BusinessInfo:
    title: str
    place_id: str
    data_id: str
    rating: float
    reviews_count: int
    address: str
    business_type: str
    coordinates: Optional[LocationData]
    phone: str = ""
    website: str = ""
    hours: str = ""
    price_level: str = ""
    distance_km: Optional[float] = None
    service_options: Dict = None

@dataclass
class ReviewData:
    place_id: str
    author: str
    rating: int
    text: str
    date: str
    relative_date: str
    likes: int = 0
    author_reviews_count: int = 0
    review_id: str = ""
    images: List[str] = None
    business_response: Dict = None

class CacheManager:
    """Simple in-memory cache for API responses"""
    
    def __init__(self, max_size: int = 100, ttl_minutes: int = 60):
        self.cache = {}
        self.max_size = max_size
        self.ttl = timedelta(minutes=ttl_minutes)
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        cache_string = f"{args}{sorted(kwargs.items())}"
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Dict]:
        """Get cached value if not expired"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Dict) -> None:
        """Set cached value with timestamp"""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[key] = (value, datetime.now())

class SentimentAnalyzer:
    """Enhanced sentiment analysis for restaurant reviews"""
    
    def __init__(self):
        # Enhanced keyword dictionaries
        self.positive_keywords = {
            'food': ['delicious', 'amazing', 'fantastic', 'excellent', 'perfect', 'outstanding', 'incredible', 'divine', 'heavenly', 'scrumptious'],
            'service': ['friendly', 'helpful', 'attentive', 'professional', 'courteous', 'efficient', 'welcoming', 'exceptional'],
            'atmosphere': ['cozy', 'charming', 'lovely', 'beautiful', 'comfortable', 'relaxing', 'peaceful', 'inviting', 'warm'],
            'general': ['love', 'recommend', 'impressed', 'satisfied', 'pleased', 'enjoyed', 'wonderful', 'great', 'good']
        }
        
        self.negative_keywords = {
            'food': ['terrible', 'awful', 'disgusting', 'bland', 'cold', 'stale', 'overcooked', 'undercooked', 'flavorless', 'inedible'],
            'service': ['rude', 'slow', 'unprofessional', 'inattentive', 'dismissive', 'unfriendly', 'poor', 'terrible'],
            'atmosphere': ['noisy', 'dirty', 'cramped', 'uncomfortable', 'chaotic', 'unwelcoming', 'dated', 'shabby'],
            'general': ['hate', 'disappointed', 'frustrated', 'angry', 'regret', 'avoid', 'waste', 'horrible', 'bad']
        }
        
        self.work_friendly_indicators = {
            'wifi': ['wifi', 'wi-fi', 'internet', 'connection', 'online', 'network'],
            'power': ['power', 'plug', 'socket', 'outlet', 'charging', 'charger', 'adapter'],
            'work_space': ['laptop', 'work', 'study', 'remote', 'meeting', 'zoom', 'quiet', 'focus', 'table space'],
            'atmosphere': ['peaceful', 'calm', 'quiet', 'relaxing', 'focused', 'concentration'],
            'negative': ['loud', 'noisy', 'crowded', 'busy', 'chaotic', 'cramped', 'limited seating']
        }
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Comprehensive sentiment analysis"""
        text_lower = text.lower()
        
        sentiment_scores = {}
        for category, keywords in self.positive_keywords.items():
            positive_count = sum(1 for keyword in keywords if keyword in text_lower)
            negative_count = sum(1 for keyword in self.negative_keywords[category] if keyword in text_lower)
            
            sentiment_scores[category] = {
                'positive': positive_count,
                'negative': negative_count,
                'net_score': positive_count - negative_count
            }
        
        # Overall sentiment
        total_positive = sum(scores['positive'] for scores in sentiment_scores.values())
        total_negative = sum(scores['negative'] for scores in sentiment_scores.values())
        
        return {
            'category_scores': sentiment_scores,
            'overall_positive': total_positive,
            'overall_negative': total_negative,
            'sentiment_score': total_positive - total_negative,
            'sentiment_label': self._get_sentiment_label(total_positive - total_negative)
        }
    
    def analyze_work_friendliness(self, text: str) -> Dict:
        """Enhanced work-friendliness analysis"""
        text_lower = text.lower()
        
        scores = {}
        for category, keywords in self.work_friendly_indicators.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[category] = score
        
        # Calculate work-friendly score
        positive_score = scores['wifi'] + scores['power'] + scores['work_space'] + scores['atmosphere']
        negative_score = scores['negative']
        
        final_score = max(0, positive_score - negative_score)
        
        if final_score >= 5:
            status = "🟢 Highly Work-Friendly"
        elif final_score >= 3:
            status = "🟡 Moderately Work-Friendly"
        elif final_score >= 1:
            status = "🟠 Limited Work Features"
        else:
            status = "🔴 Not Work-Friendly"
        
        return {
            'status': status,
            'scores': scores,
            'total_score': final_score,
            'max_score': 10,
            'percentage': min(100, (final_score / 10) * 100)
        }
    
    def _get_sentiment_label(self, score: int) -> str:
        """Convert sentiment score to label"""
        if score >= 3:
            return "Very Positive"
        elif score >= 1:
            return "Positive"
        elif score >= -1:
            return "Neutral"
        elif score >= -3:
            return "Negative"
        else:
            return "Very Negative"

class EnhancedCafeAnalyzer:
    """Enhanced cafe and restaurant analyzer with improved features"""
    
    def __init__(self, serpapi_key: str):
        logger.info("Initializing EnhancedCafeAnalyzer")
        self.serpapi_key = serpapi_key
        self.cache = CacheManager()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CafeAnalyzer/2.0 (+https://your-domain.com/bot)'
        })
        
        # Load AI model with error handling
        self.summarizer = self._load_ai_model()
    
    @st.cache_resource
    def _load_ai_model(_self):
        """Load AI summarization model with enhanced error handling"""
        try:
            logger.info("Loading AI summarization model")
            with st.spinner("🤖 Loading AI model..."):
                # Try multiple model options
                model_options = [
                    "sshleifer/distilbart-cnn-12-6",  # Primary choice
                    "facebook/bart-large-cnn",         # Fallback 1
                    "google/pegasus-xsum"              # Fallback 2
                ]
                
                for model_name in model_options:
                    try:
                        from transformers import pipeline
                        summarizer = pipeline(
                            "summarization",
                            model=model_name,
                            device=-1,  # CPU
                            framework="pt"
                        )
                        logger.info(f"Successfully loaded model: {model_name}")
                        st.success(f"✅ AI model loaded: {model_name.split('/')[-1]}")
                        return summarizer
                    except Exception as e:
                        logger.warning(f"Failed to load {model_name}: {e}")
                        continue
                
                logger.warning("All AI models failed to load")
                st.warning("⚠️ AI summarization unavailable - using fallback method")
                return None
                
        except ImportError:
            logger.error("Transformers library not available")
            st.error("❌ AI libraries not installed")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading AI model: {e}")
            st.error(f"❌ AI model error: {e}")
            return None
    
    def validate_api_key(self) -> Tuple[bool, str]:
        """Validate SerpAPI key and return account info"""
        try:
            url = "https://serpapi.com/search"
            params = {
                "engine": "google",
                "q": "test",
                "api_key": self.serpapi_key
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 401:
                return False, "Invalid API key"
            elif response.status_code == 429:
                return False, "API quota exceeded"
            elif response.status_code == 200:
                data = response.json()
                if "error" in data:
                    return False, f"API Error: {data['error']}"
                
                # Get account info if available
                search_metadata = data.get("search_metadata", {})
                return True, f"Valid API key (ID: {search_metadata.get('id', 'N/A')})"
            else:
                return False, f"HTTP {response.status_code}"
                
        except Exception as e:
            logger.error(f"API validation error: {e}")
            return False, f"Connection error: {e}"
    
    def resolve_google_maps_url(self, url: str) -> Optional[str]:
        """Enhanced Google Maps URL resolver with multiple methods"""
        try:
            logger.info(f"Resolving URL: {url[:100]}...")
            
            # Handle different URL types
            if "maps.app.goo.gl" in url or "goo.gl" in url:
                st.info("🔗 Resolving short link...")
                response = self.session.get(url, allow_redirects=True, timeout=15)
                resolved_url = response.url
                logger.info(f"Resolved to: {resolved_url[:100]}...")
                return resolved_url
            elif "google.com/maps" in url:
                return url
            else:
                logger.warning(f"Unknown URL format: {url}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"URL resolution failed: {e}")
            st.error(f"❌ Failed to resolve URL: {e}")
            return None
    
    def extract_business_info_from_url(self, google_url: str) -> Optional[Dict]:
        """Enhanced business information extraction"""
        try:
            logger.info("Extracting business info from URL")
            st.info("🔍 Analyzing Google Maps URL...")
            
            business_info = {}
            
            # Method 1: Extract from /place/ path
            if '/place/' in google_url:
                place_match = re.search(r'/place/([^/@]+)', google_url)
                if place_match:
                    encoded_name = place_match.group(1)
                    business_name = urllib.parse.unquote(encoded_name)
                    business_name = business_name.replace('+', ' ').strip()
                    business_info['name'] = business_name
            
            # Method 2: Extract from query parameters
            parsed_url = urlparse(google_url)
            query_params = parse_qs(parsed_url.query)
            
            if 'q' in query_params:
                business_info['query'] = query_params['q'][0]
            
            # Method 3: Extract coordinates
            coord_patterns = [
                r'/@(-?\d+\.\d+),(-?\d+\.\d+),\d+z',
                r'!3d(-?\d+\.\d+).*!4d(-?\d+\.\d+)',
                r'&ll=(-?\d+\.\d+),(-?\d+\.\d+)'
            ]
            
            for pattern in coord_patterns:
                match = re.search(pattern, google_url)
                if match:
                    lat, lng = float(match.group(1)), float(match.group(2))
                    business_info['coordinates'] = LocationData(
                        latitude=lat,
                        longitude=lng,
                        source="google_maps_url"
                    )
                    break
            
            # Method 4: Extract place ID if available
            place_id_match = re.search(r'place_id=([A-Za-z0-9_-]+)', google_url)
            if place_id_match:
                business_info['place_id'] = place_id_match.group(1)
            
            if business_info:
                st.success("✅ Successfully extracted business information")
                return business_info
            else:
                st.error("❌ Could not extract business information")
                return None
                
        except Exception as e:
            logger.error(f"Business info extraction failed: {e}")
            st.error(f"❌ Extraction error: {e}")
            return None
    
    def search_places(self, query: str, location: Optional[LocationData] = None, 
                     max_results: int = 20) -> Dict:
        """Enhanced place search with caching and better error handling"""
        
        # Generate cache key
        cache_key = self.cache._generate_key(query, location, max_results)
        cached_result = self.cache.get(cache_key)
        if cached_result:
            st.info("📦 Using cached results")
            return cached_result
        
        url = "https://serpapi.com/search"
        params = {
            "engine": "google_maps",
            "q": query,
            "api_key": self.serpapi_key,
            "type": "search",
            "hl": "en",
            "num": min(max_results, 20)  # SerpAPI limit
        }
        
        # Add location if provided
        if location:
            params["ll"] = f"@{location.latitude},{location.longitude},14z"
            st.info(f"🌍 Searching near ({location.latitude:.4f}, {location.longitude:.4f})")
        
        try:
            with st.spinner(f"🔍 Searching for: {query}"):
                response = self.session.get(url, params=params, timeout=30)
                
                if response.status_code == 429:
                    st.error("⏳ Rate limit exceeded. Please wait before searching again.")
                    return {"places": [], "error": "Rate limited"}
                
                if response.status_code == 401:
                    st.error("🔑 Invalid API key")
                    return {"places": [], "error": "Invalid API key"}
                
                if response.status_code != 200:
                    st.error(f"❌ API Error: HTTP {response.status_code}")
                    return {"places": [], "error": f"HTTP {response.status_code}"}
                
                data = response.json()
                
                if "error" in data:
                    st.error(f"❌ API Error: {data['error']}")
                    return {"places": [], "error": data["error"]}
                
                # Process results
                places = []
                local_results = data.get("local_results", [])
                
                for place_data in local_results:
                    gps = place_data.get("gps_coordinates", {})
                    place_location = None
                    
                    if gps.get("latitude") and gps.get("longitude"):
                        place_location = LocationData(
                            latitude=gps["latitude"],
                            longitude=gps["longitude"],
                            source="serpapi"
                        )
                    
                    # Calculate distance if user location provided
                    distance_km = None
                    if location and place_location:
                        distance_km = self._calculate_distance(
                            location.latitude, location.longitude,
                            place_location.latitude, place_location.longitude
                        )
                    
                    business = BusinessInfo(
                        title=place_data.get("title", ""),
                        place_id=place_data.get("place_id", ""),
                        data_id=place_data.get("data_id", ""),
                        rating=place_data.get("rating", 0.0),
                        reviews_count=place_data.get("reviews", 0),
                        address=place_data.get("address", ""),
                        business_type=place_data.get("type", ""),
                        coordinates=place_location,
                        phone=place_data.get("phone", ""),
                        website=place_data.get("website", ""),
                        hours=place_data.get("hours", ""),
                        price_level=place_data.get("price", ""),
                        distance_km=distance_km,
                        service_options=place_data.get("service_options", {})
                    )
                    
                    places.append(business)
                
                # Sort places intelligently
                if location and any(p.distance_km for p in places):
                    places.sort(key=lambda x: (x.distance_km or float('inf'), -x.rating))
                    st.success(f"✅ Found {len(places)} places (sorted by distance)")
                else:
                    places.sort(key=lambda x: (-x.rating, -x.reviews_count))
                    st.success(f"✅ Found {len(places)} places (sorted by rating)")
                
                result = {
                    "places": places,
                    "search_metadata": data.get("search_metadata", {}),
                    "total_found": len(places)
                }
                
                # Cache the result
                self.cache.set(cache_key, result)
                
                return result
                
        except requests.RequestException as e:
            logger.error(f"Search request failed: {e}")
            st.error(f"❌ Network error: {e}")
            return {"places": [], "error": f"Network error: {e}"}
        except Exception as e:
            logger.error(f"Unexpected search error: {e}")
            st.error(f"❌ Unexpected error: {e}")
            return {"places": [], "error": f"Unexpected error: {e}"}
    
    def fetch_reviews(self, business: BusinessInfo, max_reviews: int = 50) -> List[ReviewData]:
        """Enhanced review fetching with parallel processing"""
        
        cache_key = self.cache._generate_key(business.place_id, max_reviews)
        cached_reviews = self.cache.get(cache_key)
        if cached_reviews:
            st.info("📦 Using cached reviews")
            return [ReviewData(**review) for review in cached_reviews]
        
        logger.info(f"Fetching {max_reviews} reviews for {business.title}")
        
        all_reviews = []
        next_page_token = None
        pages_fetched = 0
        max_pages = max(1, (max_reviews + 7) // 8)  # ~8 reviews per page
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while len(all_reviews) < max_reviews and pages_fetched < max_pages:
            url = "https://serpapi.com/search"
            params = {
                "engine": "google_maps_reviews",
                "place_id": business.place_id,
                "api_key": self.serpapi_key,
                "hl": "en",
                "sort_by": "qualityScore"
            }
            
            if next_page_token:
                params["next_page_token"] = next_page_token
            
            try:
                status_text.text(f"📝 Fetching page {pages_fetched + 1}...")
                
                response = self.session.get(url, params=params, timeout=30)
                
                if response.status_code == 429:
                    st.warning("⏳ Rate limited, waiting...")
                    time.sleep(5)
                    continue
                
                if response.status_code != 200:
                    logger.error(f"Review fetch HTTP error: {response.status_code}")
                    break
                
                data = response.json()
                
                if "error" in data:
                    logger.error(f"Review fetch API error: {data['error']}")
                    break
                
                page_reviews = []
                for review_data in data.get("reviews", []):
                    review_text = (
                        review_data.get("snippet", "") or
                        review_data.get("text", "") or
                        review_data.get("extracted_snippet", {}).get("original", "")
                    )
                    
                    review = ReviewData(
                        place_id=business.place_id,
                        author=review_data.get("user", {}).get("name", "Anonymous"),
                        rating=review_data.get("rating", 0),
                        text=review_text,
                        date=review_data.get("date", ""),
                        relative_date=review_data.get("relative_date", ""),
                        likes=review_data.get("likes", 0),
                        author_reviews_count=review_data.get("user", {}).get("reviews", 0),
                        review_id=review_data.get("review_id", ""),
                        images=review_data.get("images", []),
                        business_response=review_data.get("response", {})
                    )
                    
                    page_reviews.append(review)
                
                all_reviews.extend(page_reviews)
                pages_fetched += 1
                
                # Update progress
                progress = min(len(all_reviews) / max_reviews, 1.0)
                progress_bar.progress(progress)
                
                # Check for next page
                next_page_token = data.get("search_metadata", {}).get("next_page_token")
                if not next_page_token:
                    break
                
                time.sleep(1.5)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Review fetch error: {e}")
                break
        
        progress_bar.progress(1.0)
        status_text.text(f"✅ Fetched {len(all_reviews)} reviews")
        
        # Cache the reviews
        review_dicts = [review.__dict__ for review in all_reviews]
        self.cache.set(cache_key, review_dicts)
        
        logger.info(f"Successfully fetched {len(all_reviews)} reviews")
        return all_reviews[:max_reviews]
    
    def analyze_reviews(self, reviews: List[ReviewData]) -> Dict:
        """Comprehensive review analysis"""
        if not reviews:
            return {"error": "No reviews to analyze"}
        
        logger.info(f"Analyzing {len(reviews)} reviews")
        
        analysis = {
            "total_reviews": len(reviews),
            "rating_distribution": {},
            "sentiment_analysis": {},
            "work_friendliness": {},
            "menu_items": [],
            "key_themes": {},
            "temporal_analysis": {},
            "review_quality": {}
        }
        
        # Rating distribution
        ratings = [r.rating for r in reviews if r.rating > 0]
        if ratings:
            analysis["rating_distribution"] = {
                "average": statistics.mean(ratings),
                "median": statistics.median(ratings),
                "std_dev": statistics.stdev(ratings) if len(ratings) > 1 else 0,
                "distribution": dict(Counter(ratings))
            }
        
        # Combine all review text for analysis
        combined_text = " ".join([r.text for r in reviews if r.text])
        
        # Sentiment analysis
        analysis["sentiment_analysis"] = self.sentiment_analyzer.analyze_sentiment(combined_text)
        
        # Work-friendliness analysis
        analysis["work_friendliness"] = self.sentiment_analyzer.analyze_work_friendliness(combined_text)
        
        # Extract menu items
        analysis["menu_items"] = self._extract_menu_items(combined_text)
        
        # Key themes extraction
        analysis["key_themes"] = self._extract_key_themes(reviews)
        
        # Temporal analysis
        analysis["temporal_analysis"] = self._analyze_temporal_patterns(reviews)
        
        # Review quality metrics
        analysis["review_quality"] = self._analyze_review_quality(reviews)
        
        logger.info("Review analysis completed")
        return analysis
    
    def generate_ai_summary(self, reviews: List[ReviewData], analysis: Dict) -> str:
        """Generate AI-powered summary with fallback"""
        try:
            if not self.summarizer:
                return self._generate_manual_summary(reviews, analysis)
            
            # Prepare text for summarization
            review_texts = [r.text for r in reviews if r.text.strip()]
            if not review_texts:
                return "No meaningful review content available for analysis."
            
            # Combine and limit text length
            combined_text = " ".join(review_texts)
            max_chars = 1000  # Model limitation
            
            if len(combined_text) > max_chars:
                combined_text = combined_text[:max_chars] + "..."
            
            with st.spinner("🤖 Generating AI summary..."):
                summary_result = self.summarizer(
                    combined_text,
                    max_length=150,
                    min_length=50,
                    do_sample=False,
                    truncation=True
                )
                
                ai_summary = summary_result[0]['summary_text']
                logger.info("AI summary generated successfully")
                
                # Enhance with analysis data
                enhanced_summary = self._enhance_summary_with_analysis(ai_summary, analysis)
                return enhanced_summary
                
        except Exception as e:
            logger.warning(f"AI summarization failed: {e}")
            st.warning("AI summarization failed, using manual analysis")
            return self._generate_manual_summary(reviews, analysis)
    
    def _generate_manual_summary(self, reviews: List[ReviewData], analysis: Dict) -> str:
        """Generate manual summary when AI is unavailable"""
        rating_dist = analysis.get("rating_distribution", {})
        sentiment = analysis.get("sentiment_analysis", {})
        work_friendly = analysis.get("work_friendliness", {})
        
        avg_rating = rating_dist.get("average", 0)
        sentiment_label = sentiment.get("sentiment_label", "Neutral")
        
        return f"""Based on analysis of {analysis['total_reviews']} reviews, this establishment maintains an average rating of {avg_rating:.1f}/5 stars with generally {sentiment_label.lower()} customer feedback. 

The reviews indicate varied experiences across food quality, service efficiency, and overall atmosphere. Customer sentiment analysis reveals specific strengths and areas for improvement in the dining experience.

Work-friendly rating: {work_friendly.get('status', 'Not assessed')} with a score of {work_friendly.get('total_score', 0)}/10."""
    
    def _enhance_summary_with_analysis(self, ai_summary: str, analysis: Dict) -> str:
        """Enhance AI summary with structured analysis data"""
        work_friendly = analysis.get("work_friendliness", {})
        menu_items = analysis.get("menu_items", [])
        
        enhanced = f"""🤖 **AI Analysis Summary:**
{ai_summary}

💻 **Work-Friendly Assessment:**
{work_friendly.get('status', 'Not assessed')} (Score: {work_friendly.get('total_score', 0)}/10)

🍽️ **Popular Menu Items:**
{', '.join(menu_items[:5]) if menu_items else 'No specific items frequently mentioned'}

📊 **Review Quality:** {analysis.get('review_quality', {}).get('average_length', 0):.0f} avg characters per review"""
        
        return enhanced
    
    def _extract_menu_items(self, text: str) -> List[str]:
        """Enhanced menu item extraction"""
        menu_keywords = {
            'coffee_drinks': ['espresso', 'latte', 'cappuccino', 'americano', 'macchiato', 'mocha', 'cold brew', 'frappe'],
            'tea_drinks': ['tea', 'matcha', 'chai', 'green tea', 'black tea', 'herbal tea', 'iced tea'],
            'food_items': ['sandwich', 'salad', 'soup', 'pasta', 'pizza', 'burger', 'wrap', 'panini'],
            'breakfast': ['croissant', 'bagel', 'muffin', 'toast', 'pancakes', 'waffle', 'eggs benedict', 'omelette'],
            'desserts': ['cake', 'cookie', 'brownie', 'cheesecake', 'pie', 'tart', 'ice cream', 'gelato'],
            'healthy': ['avocado toast', 'quinoa', 'kale', 'smoothie', 'acai bowl', 'organic', 'gluten free', 'vegan']
        }
        
        found_items = []
        text_lower = text.lower()
        
        for category, items in menu_keywords.items():
            for item in items:
                pattern = r'\b' + re.escape(item.lower()) + r'\b'
                matches = len(re.findall(pattern, text_lower))
                if matches > 0:
                    found_items.extend([item.title()] * matches)
        
        # Return top mentioned items
        item_counts = Counter(found_items)
        return [item for item, count in item_counts.most_common(10)]
    
    def _extract_key_themes(self, reviews: List[ReviewData]) -> Dict:
        """Extract key themes from reviews"""
        themes = {
            'service': 0,
            'food_quality': 0,
            'atmosphere': 0,
            'value': 0,
            'cleanliness': 0,
            'speed': 0
        }
        
        theme_keywords = {
            'service': ['service', 'staff', 'waiter', 'waitress', 'server', 'friendly', 'helpful', 'rude'],
            'food_quality': ['food', 'taste', 'flavor', 'delicious', 'fresh', 'quality', 'meal'],
            'atmosphere': ['atmosphere', 'ambiance', 'cozy', 'noise', 'music', 'decor', 'vibe'],
            'value': ['price', 'value', 'expensive', 'cheap', 'worth', 'cost', 'money'],
            'cleanliness': ['clean', 'dirty', 'hygiene', 'sanitized', 'spotless', 'messy'],
            'speed': ['fast', 'slow', 'quick', 'wait', 'time', 'prompt', 'delay']
        }
        
        for review in reviews:
            text_lower = review.text.lower()
            for theme, keywords in theme_keywords.items():
                mentions = sum(1 for keyword in keywords if keyword in text_lower)
                themes[theme] += mentions
        
        return themes
    
    def _analyze_temporal_patterns(self, reviews: List[ReviewData]) -> Dict:
        """Analyze temporal patterns in reviews"""
        try:
            dates = []
            ratings_by_date = []
            
            for review in reviews:
                if review.date:
                    try:
                        # Try to parse various date formats
                        date_obj = pd.to_datetime(review.date, errors='coerce')
                        if pd.notna(date_obj):
                            dates.append(date_obj)
                            ratings_by_date.append(review.rating)
                    except:
                        continue
            
            if not dates:
                return {"error": "No valid dates found"}
            
            df = pd.DataFrame({'date': dates, 'rating': ratings_by_date})
            df = df.sort_values('date')
            
            # Monthly aggregation
            df['month'] = df['date'].dt.to_period('M')
            monthly_stats = df.groupby('month').agg({
                'rating': ['mean', 'count']
            }).round(2)
            
            return {
                "date_range": {
                    "earliest": str(min(dates).date()),
                    "latest": str(max(dates).date())
                },
                "monthly_trends": monthly_stats.to_dict(),
                "total_days_span": (max(dates) - min(dates)).days
            }
            
        except Exception as e:
            logger.warning(f"Temporal analysis failed: {e}")
            return {"error": "Temporal analysis failed"}
    
    def _analyze_review_quality(self, reviews: List[ReviewData]) -> Dict:
        """Analyze review quality metrics"""
        if not reviews:
            return {}
        
        review_lengths = [len(review.text) for review in reviews if review.text]
        review_likes = [review.likes for review in reviews]
        
        return {
            "average_length": statistics.mean(review_lengths) if review_lengths else 0,
            "median_length": statistics.median(review_lengths) if review_lengths else 0,
            "total_likes": sum(review_likes),
            "average_likes": statistics.mean(review_likes) if review_likes else 0,
            "detailed_reviews": len([r for r in reviews if len(r.text) > 100]),
            "short_reviews": len([r for r in reviews if len(r.text) <= 50])
        }
    
    def _calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate distance using Haversine formula"""
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = math.radians(lat1)
        lng1_rad = math.radians(lng1)
        lat2_rad = math.radians(lat2)
        lng2_rad = math.radians(lng2)
        
        dlat = lat2_rad - lat1_rad
        dlng = lng2_rad - lng1_rad
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        return round(R * c, 2)

class VisualizationManager:
    """Enhanced visualization manager for better charts"""
    
    @staticmethod
    def create_rating_distribution_chart(analysis: Dict) -> go.Figure:
        """Create enhanced rating distribution chart"""
        rating_dist = analysis.get("rating_distribution", {}).get("distribution", {})
        
        if not rating_dist:
            return None
        
        ratings = list(rating_dist.keys())
        counts = list(rating_dist.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=ratings,
                y=counts,
                marker_color=['#ff4444' if r <= 2 else '#ffa500' if r == 3 else '#90EE90' for r in ratings],
                text=counts,
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="⭐ Rating Distribution",
            xaxis_title="Star Rating",
            yaxis_title="Number of Reviews",
            showlegend=False,
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_sentiment_radar_chart(analysis: Dict) -> go.Figure:
        """Create sentiment radar chart"""
        sentiment_data = analysis.get("sentiment_analysis", {}).get("category_scores", {})
        
        if not sentiment_data:
            return None
        
        categories = list(sentiment_data.keys())
        scores = [data.get("net_score", 0) for data in sentiment_data.values()]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=categories,
            fill='toself',
            name='Sentiment Score',
            line_color='rgb(32, 201, 151)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[-5, 5])
            ),
            showlegend=False,
            title="📊 Sentiment Analysis by Category",
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_menu_popularity_chart(menu_items: List[str]) -> go.Figure:
        """Create menu item popularity chart"""
        if not menu_items:
            return None
        
        item_counts = Counter(menu_items)
        top_items = dict(item_counts.most_common(10))
        
        if not top_items:
            return None
        
        fig = go.Figure(data=[
            go.Bar(
                y=list(top_items.keys()),
                x=list(top_items.values()),
                orientation='h',
                marker_color='viridis',
                text=list(top_items.values()),
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="🍽️ Most Mentioned Menu Items",
            xaxis_title="Mentions in Reviews",
            yaxis_title="Menu Items",
            height=400,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_work_friendly_gauge(analysis: Dict) -> go.Figure:
        """Create work-friendliness gauge chart"""
        work_data = analysis.get("work_friendliness", {})
        score = work_data.get("total_score", 0)
        percentage = work_data.get("percentage", 0)
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = percentage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "💻 Work-Friendly Score"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgray"},
                    {'range': [25, 50], 'color': "gray"},
                    {'range': [50, 75], 'color': "lightgreen"},
                    {'range': [75, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=400)
        return fig

class InteractiveMap:
    """Enhanced interactive map functionality"""
    
    @staticmethod
    def create_business_map(business: BusinessInfo, user_location: Optional[LocationData] = None) -> folium.Map:
        """Create enhanced interactive map"""
        if not business.coordinates:
            return None
        
        # Center map on business
        center_lat = business.coordinates.latitude
        center_lng = business.coordinates.longitude
        
        # Create map with better tiles
        m = folium.Map(
            location=[center_lat, center_lng],
            zoom_start=15,
            tiles='OpenStreetMap'
        )
        
        # Add business marker with detailed popup
        popup_html = f"""
        <div style="width: 300px;">
            <h4>☕ {business.title}</h4>
            <p><strong>📍 Address:</strong> {business.address}</p>
            <p><strong>⭐ Rating:</strong> {business.rating}/5 ({business.reviews_count} reviews)</p>
            <p><strong>📞 Phone:</strong> {business.phone or 'N/A'}</p>
            <p><strong>🌐 Website:</strong> <a href="{business.website}" target="_blank">Visit</a></p>
            <p><strong>💰 Price:</strong> {business.price_level or 'N/A'}</p>
        </div>
        """
        
        folium.Marker(
            [center_lat, center_lng],
            popup=folium.Popup(popup_html, max_width=350),
            tooltip=f"☕ {business.title}",
            icon=folium.Icon(color='red', icon='coffee', prefix='fa')
        ).add_to(m)
        
        # Add user location if available
        if user_location:
            folium.Marker(
                [user_location.latitude, user_location.longitude],
                popup="📍 Your Location",
                tooltip="You are here",
                icon=folium.Icon(color='blue', icon='user', prefix='fa')
            ).add_to(m)
            
            # Add distance line
            folium.PolyLine(
                locations=[
                    [user_location.latitude, user_location.longitude],
                    [center_lat, center_lng]
                ],
                weight=3,
                color='blue',
                opacity=0.7,
                popup=f"Distance: {business.distance_km:.2f} km"
            ).add_to(m)
            
            # Fit bounds to show both locations
            m.fit_bounds([
                [user_location.latitude, user_location.longitude],
                [center_lat, center_lng]
            ])
        
        # Add scale and fullscreen controls
        folium.plugins.MeasureControl().add_to(m)
        folium.plugins.Fullscreen().add_to(m)
        
        return m

def main():
    """Enhanced main application"""
    logger.info("Starting Enhanced Cafe & Restaurant Analyzer")
    
    # Custom CSS for better styling
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
    
    # Enhanced header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Smart Cafe & Restaurant Analyzer</h1>
        <p>AI-powered analysis with advanced insights and work-friendliness detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check for API key
    api_key = os.getenv("SERPAPI", "")
    if not api_key:
        st.error("🔑 **Configuration Error**: SerpAPI key not found in environment variables.")
        st.markdown("""
        **For Administrators**: Please set the SERPAPI environment variable:
        ```bash
        export SERPAPI=your_serpapi_key_here
        ```
        """)
        st.stop()
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = EnhancedCafeAnalyzer(api_key)
    
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    
    if 'selected_business' not in st.session_state:
        st.session_state.selected_business = None
    
    if 'reviews' not in st.session_state:
        st.session_state.reviews = None
    
    if 'analysis' not in st.session_state:
        st.session_state.analysis = None
    
    analyzer = st.session_state.analyzer
    
    # Validate API key on startup
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        with st.expander("🔑 API Status", expanded=False):
            is_valid, message = analyzer.validate_api_key()
            if is_valid:
                st.success(f"✅ {message}")
            else:
                st.error(f"❌ {message}")
        
        # Enhanced settings
        st.subheader("📊 Analysis Settings")
        max_reviews = st.slider("Number of reviews to analyze", 10, 100, 30, 5)
        enable_ai = st.checkbox("Enable AI summarization", value=True)
        show_advanced = st.checkbox("Show advanced metrics", value=False)
        
        # Location settings
        st.subheader("📍 Location Settings")
        
        # Initialize location state
        if 'user_location' not in st.session_state:
            st.session_state.user_location = None
        
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
    
    # Main search interface
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
    
    # Process search
    if search_button:
        if search_method == "🏪 Business Name" and search_query:
            logger.info(f"Searching by name: {search_query}")
            
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
            logger.info("Processing Google Maps URL")
            
            # Resolve URL if needed
            resolved_url = analyzer.resolve_google_maps_url(google_url)
            if not resolved_url:
                st.error("❌ Failed to resolve Google Maps URL")
                st.stop()
            
            # Extract business info
            business_info = analyzer.extract_business_info_from_url(resolved_url)
            if not business_info:
                st.error("❌ Could not extract business information from URL")
                st.stop()
            
            # Search using extracted info
            search_term = business_info.get('name') or business_info.get('query', '')
            if not search_term:
                st.error("❌ Could not determine search term from URL")
                st.stop()
            
            st.info(f"🔍 Searching for: **{search_term}**")
            
            user_location = st.session_state.user_location
            if not user_location and business_info.get('coordinates'):
                user_location = business_info['coordinates']
            
            results = analyzer.search_places(search_term, user_location)
            # --- Place ID mismatch diagnostics ---
            extracted_place_id = business_info.get('place_id')
            found_place_ids = [b.place_id for b in results.get('places', [])]
            if extracted_place_id:
                if extracted_place_id not in found_place_ids:
                    logger.warning(f"Extracted place_id {extracted_place_id} not found in SerpAPI search results. Found: {found_place_ids}")
                    st.warning(f"⚠️ The place_id extracted from the URL ({extracted_place_id}) was not found in the search results.\nThis may indicate a mismatch between Google Maps and SerpAPI data.\nTry searching by business name or check if the business is new or renamed.")
                else:
                    logger.info(f"Extracted place_id {extracted_place_id} matches a result from SerpAPI.")
            # --- End diagnostics ---

            if "error" in results:
                st.error(f"❌ Search failed: {results['error']}")
            elif not results.get("places"):
                st.error(f"❌ No results found for extracted business: {search_term}")
            else:
                st.session_state.search_results = results
                st.success(f"✅ Found {len(results['places'])} matching businesses")
        else:
            st.error("❌ Please provide either a business name or Google Maps URL")
    
    # Display search results
    if st.session_state.search_results:
        st.markdown("---")
        st.header("🏪 Select Your Business")
        
        places = st.session_state.search_results["places"]
        
        if len(places) == 1:
            # Auto-select single result
            st.session_state.selected_business = places[0]
            business = places[0]
            st.success(f"✅ Auto-selected: **{business.title}**")
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
            
            # Advanced metrics (if enabled)
            if show_advanced:
                st.subheader("🔬 Advanced Analytics")
                
                # Temporal analysis
                temporal_data = analysis.get("temporal_analysis", {})
                if "error" not in temporal_data:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**📅 Review Timeline:**")
                        date_range = temporal_data.get("date_range", {})
                        st.markdown(f"• Earliest: {date_range.get('earliest', 'N/A')}")
                        st.markdown(f"• Latest: {date_range.get('latest', 'N/A')}")
                        st.markdown(f"• Span: {temporal_data.get('total_days_span', 0)} days")
                    
                    with col2:
                        st.markdown("**📊 Review Quality:**")
                        quality = analysis.get("review_quality", {})
                        st.markdown(f"• Avg Length: {quality.get('average_length', 0):.0f} chars")
                        st.markdown(f"• Total Likes: {quality.get('total_likes', 0)}")
                        st.markdown(f"• Detailed Reviews: {quality.get('detailed_reviews', 0)}")
        
        with tabs[2]:  # Location & Map
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
        
        with tabs[3]:  # Work Analysis
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
                percentage = work_data.get("percentage", 0)
                
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
            
            # Detailed breakdown
            st.markdown("**🔍 Detailed Analysis:**")
            
            feature_details = {
                'wifi': 'WiFi and internet connectivity mentions',
                'power': 'Power outlets and charging stations',
                'work_space': 'Laptop-friendly tables and work mentions',
                'atmosphere': 'Quiet, peaceful environment for focus',
                'negative': 'Noise and crowding concerns'
            }
            
            for feature, description in feature_details.items():
                score = scores.get(feature, 0)
                if score > 0:
                    st.markdown(f"• **{description}:** {score} mentions in reviews")
        
        with tabs[4]:  # Review Explorer
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
            for i, review in enumerate(filtered_reviews, 1):
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
                        review_sentiment = analyzer.sentiment_analyzer.analyze_sentiment(review.text)
                        sentiment_score = review_sentiment.get("sentiment_score", 0)
                        
                        if sentiment_score > 0:
                            st.success(f"😊 Positive ({sentiment_score})")
                        elif sentiment_score < 0:
                            st.error(f"😞 Negative ({sentiment_score})")
                        else:
                            st.info("😐 Neutral")
                        
                        # Show review images if available
                        if review.images:
                            st.markdown("**📸 Images:**")
                            for img_url in review.images[:2]:  # Limit to 2 images
                                try:
                                    st.image(img_url, width=100)
                                except:
                                    st.markdown(f"[Image]({img_url})")
        
        with tabs[5]:  # Export Data
            st.subheader("📥 Export & Download Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Data Export Options:**")
                
                # CSV Export
                if st.button("📊 Download Reviews CSV", use_container_width=True):
                    df_data = []
                    for review in reviews:
                        df_data.append({
                            'business_name': business.title,
                            'author': review.author,
                            'rating': review.rating,
                            'date': review.date,
                            'text': review.text,
                            'likes': review.likes,
                            'author_reviews_count': review.author_reviews_count
                        })
                    
                    df = pd.DataFrame(df_data)
                    csv = df.to_csv(index=False)
                    
                    st.download_button(
                        label="💾 Download CSV File",
                        data=csv,
                        file_name=f"{business.title.replace(' ', '_')}_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                # JSON Export
                if st.button("🔧 Download Complete Data (JSON)", use_container_width=True):
                    export_data = {
                        "business_info": {
                            "title": business.title,
                            "address": business.address,
                            "rating": business.rating,
                            "reviews_count": business.reviews_count,
                            "phone": business.phone,
                            "website": business.website,
                            "coordinates": {
                                "latitude": business.coordinates.latitude if business.coordinates else None,
                                "longitude": business.coordinates.longitude if business.coordinates else None
                            } if business.coordinates else None
                        },
                        "reviews": [review.__dict__ for review in reviews],
                        "analysis": analysis,
                        "export_metadata": {
                            "generated_at": datetime.now().isoformat(),
                            "total_reviews_analyzed": len(reviews),
                            "analysis_version": "2.0"
                        }
                    }
                    
                    json_str = json.dumps(export_data, indent=2, ensure_ascii=False, default=str)
                    
                    st.download_button(
                        label="💾 Download JSON File",
                        data=json_str,
                        file_name=f"{business.title.replace(' ', '_')}_complete_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            with col2:
                st.markdown("**📄 Report Generation:**")
                
                # Comprehensive Report
                if st.button("📋 Generate Analysis Report", use_container_width=True):
                    rating_avg = analysis.get("rating_distribution", {}).get("average", 0)
                    work_data = analysis.get("work_friendliness", {})
                    sentiment_data = analysis.get("sentiment_analysis", {})
                    
                    report = f"""
# 🎯 COMPREHENSIVE CAFE/RESTAURANT ANALYSIS REPORT

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🏪 BUSINESS INFORMATION
- **Name:** {business.title}
- **Address:** {business.address}
- **Phone:** {business.phone or 'Not available'}
- **Website:** {business.website or 'Not available'}
- **Business Type:** {business.business_type}
- **Price Level:** {business.price_level or 'Not specified'}

## 📊 ANALYSIS SUMMARY
- **Reviews Analyzed:** {len(reviews)}
- **Average Rating:** {rating_avg:.1f}/5.0
- **Overall Sentiment:** {sentiment_data.get('sentiment_label', 'Neutral')}
- **Work-Friendly Score:** {work_data.get('total_score', 0)}/10
- **Work-Friendly Status:** {work_data.get('status', 'Not assessed')}

## 🍽️ POPULAR MENU ITEMS
{chr(10).join([f"- {item}" for item in analysis.get('menu_items', [])[:10]]) or "- No specific items frequently mentioned"}

## 💻 WORK-FRIENDLINESS ANALYSIS
**Overall Status:** {work_data.get('status', 'Not assessed')}
**Score Breakdown:**
{chr(10).join([f"- {k.replace('_', ' ').title()}: {v} mentions" for k, v in work_data.get('scores', {}).items()])}

## 📈 KEY INSIGHTS
**Top Themes Mentioned:**
{chr(10).join([f"- {k.replace('_', ' ').title()}: {v} mentions" for k, v in sorted(analysis.get('key_themes', {}).items(), key=lambda x: x[1], reverse=True)[:5]])}

**Sentiment Breakdown:**
{chr(10).join([f"- {k.replace('_', ' ').title()}: {v.get('net_score', 0)} net score" for k, v in sentiment_data.get('category_scores', {}).items()])}

## 📝 REVIEW QUALITY METRICS
- **Average Review Length:** {analysis.get('review_quality', {}).get('average_length', 0):.0f} characters
- **Total Likes Received:** {analysis.get('review_quality', {}).get('total_likes', 0)}
- **Detailed Reviews (>100 chars):** {analysis.get('review_quality', {}).get('detailed_reviews', 0)}

---
*Report generated by Smart Cafe & Restaurant Analyzer v2.0*
*Data source: Google Reviews via SerpAPI*
"""
                    
                    st.download_button(
                        label="💾 Download Analysis Report",
                        data=report,
                        file_name=f"{business.title.replace(' ', '_')}_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                
                # Social Media Summary
                if st.button("📱 Generate Social Share Summary", use_container_width=True):
                    share_text = f"""☕ {business.title} - Analysis Summary

⭐ Rating: {rating_avg:.1f}/5 ({len(reviews)} reviews analyzed)
📍 {business.address}

🎯 Key Highlights:
• Overall sentiment: {sentiment_data.get('sentiment_label', 'Neutral')}
• Work-friendly score: {work_data.get('total_score', 0)}/10
• Popular items: {', '.join(analysis.get('menu_items', [])[:3]) or 'Various menu options'}

💻 {work_data.get('status', 'Work-friendliness not assessed')}

#CafeReview #RestaurantAnalysis #DataDriven #SmartDining"""
                    
                    st.text_area(
                        "📋 Copy this summary to share:",
                        share_text,
                        height=200,
                        key="share_summary"
                    )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🎯 <strong>Smart Cafe & Restaurant Analyzer v2.0</strong></p>
        <p>Powered by AI • Enhanced Analytics • Real-time Data</p>
        <p><small>Data sourced from Google Reviews via SerpAPI</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    logger.info("Starting application as main module")
    
    # Clear cache on restart if needed
    if hasattr(st, 'cache_resource'):
        # Uncomment to clear cache on restart
        # st.cache_resource.clear()
        pass
    
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error(f"❌ Application error: {e}")
        st.markdown("Please refresh the page or contact support if the issue persists.")