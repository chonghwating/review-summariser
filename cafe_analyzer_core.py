"""
Core business logic for the Cafe & Restaurant Analyzer
Separated from Streamlit UI for better maintainability and testing
"""

import requests
import json
import time
import re
import urllib.parse
from typing import List, Dict, Optional, Tuple, Union
import pandas as pd
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from functools import lru_cache
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from urllib.parse import urlparse, parse_qs
from collections import Counter, defaultdict
import math
import statistics
from dataclasses import dataclass

# Configure logging
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
    
    def _load_ai_model(self):
        """Load AI summarization model with enhanced error handling"""
        try:
            logger.info("Loading AI summarization model")
            # Try multiple model options
            model_options = [
                "philschmid/bart-large-cnn-samsum" 
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
                    return summarizer
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {e}")
                    continue
            
            logger.warning("All AI models failed to load")
            return None
            
        except ImportError:
            logger.error("Transformers library not available")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading AI model: {e}")
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
            return None
    
    def extract_business_info_from_url(self, google_url: str) -> Optional[Dict]:
        """Enhanced business information extraction"""
        try:
            logger.info("Extracting business info from URL")
            
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
            
            return business_info if business_info else None
                
        except Exception as e:
            logger.error(f"Business info extraction failed: {e}")
            return None
    
    def search_places(self, query: str, location: Optional[LocationData] = None, 
                     max_results: int = 20) -> Dict:
        """Enhanced place search with caching and better error handling"""
        
        # Generate cache key
        cache_key = self.cache._generate_key(query, location, max_results)
        cached_result = self.cache.get(cache_key)
        if cached_result:
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
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 429:
                return {"places": [], "error": "Rate limited"}
            
            if response.status_code == 401:
                return {"places": [], "error": "Invalid API key"}
            
            if response.status_code != 200:
                return {"places": [], "error": f"HTTP {response.status_code}"}
            
            data = response.json()
            
            if "error" in data:
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
            else:
                places.sort(key=lambda x: (-x.rating, -x.reviews_count))
            
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
            return {"places": [], "error": f"Network error: {e}"}
        except Exception as e:
            logger.error(f"Unexpected search error: {e}")
            return {"places": [], "error": f"Unexpected error: {e}"}
    
    def fetch_reviews(self, business: BusinessInfo, max_reviews: int = 50, progress_callback=None) -> List[ReviewData]:
        """Enhanced review fetching with progress callback and detailed logging"""
        
        cache_key = self.cache._generate_key(business.place_id, max_reviews)
        cached_reviews = self.cache.get(cache_key)
        if cached_reviews:
            logger.info(f"📦 Found cached reviews for {business.title}")
            if progress_callback:
                progress_callback(1.0, f"✅ Loaded {len(cached_reviews)} cached reviews")
            return [ReviewData(**review) for review in cached_reviews]
        
        logger.info(f"🔍 Starting to fetch {max_reviews} reviews for '{business.title}' (place_id: {business.place_id})")
        
        all_reviews = []
        next_page_token = None
        pages_fetched = 0
        max_pages = max(1, (max_reviews + 7) // 8)  # ~8 reviews per page
        
        logger.info(f"📄 Will fetch up to {max_pages} pages to get {max_reviews} reviews")
        
        while len(all_reviews) < max_reviews and pages_fetched < max_pages:
            current_progress = pages_fetched / max_pages
            status_msg = f"📑 Fetching page {pages_fetched + 1}/{max_pages} ({len(all_reviews)} reviews collected)"
            
            if progress_callback:
                progress_callback(current_progress, status_msg)
            
            logger.info(f"📑 Fetching page {pages_fetched + 1}/{max_pages} (collected {len(all_reviews)} reviews so far)")
            
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
                logger.debug(f"🔗 Using next_page_token: {next_page_token[:20]}...")
            
            try:
                logger.debug(f"🌐 Making API request to SerpAPI...")
                response = self.session.get(url, params=params, timeout=30)
                
                logger.debug(f"📡 API Response: HTTP {response.status_code}")
                
                if response.status_code == 429:
                    logger.warning("⏳ Rate limited by API, waiting 5 seconds...")
                    if progress_callback:
                        progress_callback(current_progress, "⏳ Rate limited, waiting...")
                    time.sleep(5)
                    continue
                
                if response.status_code != 200:
                    logger.error(f"❌ HTTP error {response.status_code}: {response.text[:200]}")
                    break
                
                data = response.json()
                logger.debug(f"📊 API returned {len(data.get('reviews', []))} reviews")
                
                if "error" in data:
                    logger.error(f"❌ API error: {data['error']}")
                    break
                
                page_reviews = []
                for i, review_data in enumerate(data.get("reviews", [])):
                    logger.debug(f"📝 Processing review {i+1} from {review_data.get('user', {}).get('name', 'Anonymous')}")
                    
                    review_text = (
                        review_data.get("snippet", "") or
                        review_data.get("text", "") or
                        review_data.get("extracted_snippet", {}).get("original", "")
                    )
                    
                    if not review_text:
                        logger.warning(f"⚠️ Review {i+1} has no text content, skipping")
                        continue
                    
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
                    logger.debug(f"✅ Successfully processed review from {review.author} ({review.rating}⭐, {len(review.text)} chars)")
                
                all_reviews.extend(page_reviews)
                pages_fetched += 1
                
                logger.info(f"📋 Page {pages_fetched} complete: +{len(page_reviews)} reviews (total: {len(all_reviews)})")
                
                # Check for next page
                next_page_token = data.get("search_metadata", {}).get("next_page_token")
                if not next_page_token:
                    logger.info("🏁 No more pages available")
                    break
                
                logger.debug("⏱️ Waiting 1.5s between requests for rate limiting")
                if progress_callback:
                    progress_callback(current_progress, f"⏱️ Waiting between requests... ({len(all_reviews)} reviews so far)")
                time.sleep(1.5)  # Rate limiting
                
            except requests.RequestException as e:
                logger.error(f"🌐 Network error on page {pages_fetched + 1}: {e}")
                if progress_callback:
                    progress_callback(current_progress, f"❌ Network error: {str(e)}")
                break
            except Exception as e:
                logger.error(f"❌ Unexpected error on page {pages_fetched + 1}: {e}")
                if progress_callback:
                    progress_callback(current_progress, f"❌ Error: {str(e)}")
                break
        
        # Cache the reviews
        logger.info(f"💾 Caching {len(all_reviews)} reviews for future use")
        review_dicts = [review.__dict__ for review in all_reviews]
        self.cache.set(cache_key, review_dicts)
        
        final_reviews = all_reviews[:max_reviews]
        logger.info(f"✅ Successfully fetched {len(final_reviews)} reviews for '{business.title}'")
        
        # Log review statistics
        if final_reviews:
            avg_rating = sum(r.rating for r in final_reviews) / len(final_reviews)
            avg_length = sum(len(r.text) for r in final_reviews) / len(final_reviews)
            logger.info(f"📊 Review stats - Avg rating: {avg_rating:.1f}⭐, Avg length: {avg_length:.0f} chars")
        
        if progress_callback:
            progress_callback(1.0, f"✅ Completed! Fetched {len(final_reviews)} reviews")
        
        return final_reviews
    
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
        
        return m

class ReportGenerator:
    """Generate various types of reports and exports"""
    
    @staticmethod
    def generate_comprehensive_report(business: BusinessInfo, reviews: List[ReviewData], analysis: Dict) -> str:
        """Generate comprehensive analysis report"""
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
        return report
    
    @staticmethod
    def generate_social_share_summary(business: BusinessInfo, analysis: Dict, reviews: List[ReviewData]) -> str:
        """Generate social media share summary"""
        rating_avg = analysis.get("rating_distribution", {}).get("average", 0)
        work_data = analysis.get("work_friendliness", {})
        sentiment_data = analysis.get("sentiment_analysis", {})
        
        share_text = f"""☕ {business.title} - Analysis Summary

⭐ Rating: {rating_avg:.1f}/5 ({len(reviews)} reviews analyzed)
📍 {business.address}

🎯 Key Highlights:
• Overall sentiment: {sentiment_data.get('sentiment_label', 'Neutral')}
• Work-friendly score: {work_data.get('total_score', 0)}/10
• Popular items: {', '.join(analysis.get('menu_items', [])[:3]) or 'Various menu options'}

💻 {work_data.get('status', 'Work-friendliness not assessed')}

#CafeReview #RestaurantAnalysis #DataDriven #SmartDining"""
        
        return share_text
    
    @staticmethod
    def generate_csv_data(business: BusinessInfo, reviews: List[ReviewData]) -> str:
        """Generate CSV data for export"""
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
        return df.to_csv(index=False)
    
    @staticmethod
    def generate_json_export(business: BusinessInfo, reviews: List[ReviewData], analysis: Dict) -> str:
        """Generate complete JSON export"""
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
        
        return json.dumps(export_data, indent=2, ensure_ascii=False, default=str)