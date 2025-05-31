import requests
import os
import json
from typing import Dict, Optional
import time

class NutritionAPI:
    """Interface for fetching nutritional data from USDA FoodData Central API"""
    
    def __init__(self):
        self.api_key = os.getenv("USDA_API_KEY", "DEMO_KEY")
        self.base_url = "https://api.nal.usda.gov/fdc/v1"
        self.cache = {}  # Simple in-memory cache
        
        # Food name mappings for better API queries
        self.food_mappings = self._load_food_mappings()
    
    def _load_food_mappings(self):
        """Load food name mappings for better API queries"""
        return {
            # Common food mappings
            "apple": "apple raw",
            "banana": "banana raw",
            "orange": "orange raw",
            "pizza": "pizza cheese regular crust",
            "burger": "hamburger single patty plain",
            "rice": "rice white cooked",
            "chicken": "chicken breast meat only cooked roasted",
            "bread": "bread whole wheat",
            "egg": "egg whole raw",
            "milk": "milk whole",
            "pasta": "pasta cooked enriched",
            "potato": "potato baked flesh and skin",
            "tomato": "tomato red ripe raw",
            "carrot": "carrot raw",
            "broccoli": "broccoli raw",
            "salmon": "salmon atlantic farmed cooked",
            "yogurt": "yogurt plain whole milk",
            "cheese": "cheese cheddar",
            "avocado": "avocado raw",
            "spinach": "spinach raw"
        }
    
    def get_nutrition_info(self, food_name: str) -> Optional[Dict]:
        """Get nutritional information for a food item"""
        try:
            # Check cache first
            cache_key = food_name.lower().strip()
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Map food name if available
            search_term = self.food_mappings.get(cache_key, food_name)
            
            # Search for food in USDA database
            food_data = self._search_food(search_term)
            
            if food_data:
                nutrition_info = self._extract_nutrition_data(food_data)
                
                # Cache the result
                self.cache[cache_key] = nutrition_info
                
                return nutrition_info
            
            # Fallback to estimated values if API fails
            return self._get_fallback_nutrition(food_name)
            
        except Exception as e:
            print(f"Nutrition API error: {e}")
            return self._get_fallback_nutrition(food_name)
    
    def _search_food(self, search_term: str) -> Optional[Dict]:
        """Search for food in USDA database"""
        try:
            # Use foods/search endpoint
            search_url = f"{self.base_url}/foods/search"
            
            params = {
                "query": search_term,
                "dataType": ["Foundation", "SR Legacy"],
                "pageSize": 1,
                "api_key": self.api_key
            }
            
            response = requests.get(search_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                foods = data.get("foods", [])
                
                if foods:
                    # Get detailed info for the first result
                    food_id = foods[0]["fdcId"]
                    return self._get_food_details(food_id)
            
            return None
            
        except Exception as e:
            print(f"Food search error: {e}")
            return None
    
    def _get_food_details(self, food_id: int) -> Optional[Dict]:
        """Get detailed nutritional information for a specific food ID"""
        try:
            detail_url = f"{self.base_url}/food/{food_id}"
            
            params = {
                "api_key": self.api_key
            }
            
            response = requests.get(detail_url, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            
            return None
            
        except Exception as e:
            print(f"Food details error: {e}")
            return None
    
    def _extract_nutrition_data(self, food_data: Dict) -> Dict:
        """Extract relevant nutrition data from USDA response"""
        nutrition_info = {
            "calories": 0,
            "protein": 0,
            "fat": 0,
            "carbs": 0,
            "fiber": 0,
            "sugar": 0
        }
        
        try:
            nutrients = food_data.get("foodNutrients", [])
            
            for nutrient in nutrients:
                nutrient_name = nutrient.get("nutrient", {}).get("name", "").lower()
                amount = nutrient.get("amount", 0)
                
                if "energy" in nutrient_name and "kcal" in nutrient_name:
                    nutrition_info["calories"] = amount
                elif "protein" in nutrient_name:
                    nutrition_info["protein"] = amount
                elif "total lipid" in nutrient_name or "fat" in nutrient_name:
                    nutrition_info["fat"] = amount
                elif "carbohydrate" in nutrient_name and "by difference" in nutrient_name:
                    nutrition_info["carbs"] = amount
                elif "fiber" in nutrient_name:
                    nutrition_info["fiber"] = amount
                elif "sugars" in nutrient_name and "total" in nutrient_name:
                    nutrition_info["sugar"] = amount
            
            return nutrition_info
            
        except Exception as e:
            print(f"Nutrition extraction error: {e}")
            return nutrition_info
    
    def _get_fallback_nutrition(self, food_name: str) -> Dict:
        """Provide estimated nutritional values when API is unavailable"""
        
        # Common food nutrition estimates (per 100g)
        fallback_data = {
            "apple": {"calories": 52, "protein": 0.3, "fat": 0.2, "carbs": 14, "fiber": 2.4, "sugar": 10},
            "banana": {"calories": 89, "protein": 1.1, "fat": 0.3, "carbs": 23, "fiber": 2.6, "sugar": 12},
            "orange": {"calories": 47, "protein": 0.9, "fat": 0.1, "carbs": 12, "fiber": 2.4, "sugar": 9},
            "pizza": {"calories": 266, "protein": 11, "fat": 10, "carbs": 33, "fiber": 2, "sugar": 4},
            "burger": {"calories": 295, "protein": 17, "fat": 14, "carbs": 27, "fiber": 2, "sugar": 4},
            "rice": {"calories": 130, "protein": 2.7, "fat": 0.3, "carbs": 28, "fiber": 0.4, "sugar": 0},
            "chicken": {"calories": 165, "protein": 31, "fat": 3.6, "carbs": 0, "fiber": 0, "sugar": 0},
            "bread": {"calories": 247, "protein": 13, "fat": 4.2, "carbs": 41, "fiber": 6, "sugar": 5},
            "egg": {"calories": 155, "protein": 13, "fat": 11, "carbs": 1.1, "fiber": 0, "sugar": 1.1}
        }
        
        food_key = food_name.lower().strip()
        
        # Try exact match first
        if food_key in fallback_data:
            return fallback_data[food_key]
        
        # Try partial match
        for key, data in fallback_data.items():
            if key in food_key or food_key in key:
                return data
        
        # Default values for unknown foods
        return {
            "calories": 200,
            "protein": 5,
            "fat": 8,
            "carbs": 25,
            "fiber": 3,
            "sugar": 5
        }
    
    def validate_api_key(self) -> bool:
        """Validate if the API key is working"""
        try:
            test_url = f"{self.base_url}/foods/search"
            params = {
                "query": "apple",
                "pageSize": 1,
                "api_key": self.api_key
            }
            
            response = requests.get(test_url, params=params, timeout=5)
            return response.status_code == 200
            
        except Exception:
            return False
