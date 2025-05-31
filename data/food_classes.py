"""
Food classification classes based on Food-101 dataset
This list contains 101 common food categories for classification
"""

FOOD_CLASSES = [
    "apple_pie",
    "baby_back_ribs",
    "baklava",
    "beef_carpaccio",
    "beef_tartare",
    "beet_salad",
    "beignets",
    "bibimbap",
    "bread_pudding",
    "breakfast_burrito",
    "bruschetta",
    "caesar_salad",
    "cannoli",
    "caprese_salad",
    "carrot_cake",
    "ceviche",
    "cheese_plate",
    "cheesecake",
    "chicken_curry",
    "chicken_quesadilla",
    "chicken_wings",
    "chocolate_cake",
    "chocolate_mousse",
    "churros",
    "clam_chowder",
    "club_sandwich",
    "crab_cakes",
    "creme_brulee",
    "croque_madame",
    "cup_cakes",
    "deviled_eggs",
    "donuts",
    "dumplings",
    "edamame",
    "eggs_benedict",
    "escargots",
    "falafel",
    "filet_mignon",
    "fish_and_chips",
    "foie_gras",
    "french_fries",
    "french_onion_soup",
    "french_toast",
    "fried_calamari",
    "fried_rice",
    "frozen_yogurt",
    "garlic_bread",
    "gnocchi",
    "greek_salad",
    "grilled_cheese_sandwich",
    "grilled_salmon",
    "guacamole",
    "gyoza",
    "hamburger",
    "hot_and_sour_soup",
    "hot_dog",
    "huevos_rancheros",
    "hummus",
    "ice_cream",
    "lasagna",
    "lobster_bisque",
    "lobster_roll_sandwich",
    "macaroni_and_cheese",
    "macarons",
    "miso_soup",
    "mussels",
    "nachos",
    "omelette",
    "onion_rings",
    "oysters",
    "pad_thai",
    "paella",
    "pancakes",
    "panna_cotta",
    "peking_duck",
    "pho",
    "pizza",
    "pork_chop",
    "poutine",
    "prime_rib",
    "pulled_pork_sandwich",
    "ramen",
    "ravioli",
    "red_velvet_cake",
    "risotto",
    "samosa",
    "sashimi",
    "scallops",
    "seaweed_salad",
    "shrimp_and_grits",
    "spaghetti_bolognese",
    "spaghetti_carbonara",
    "spring_rolls",
    "steak",
    "strawberry_shortcake",
    "sushi",
    "tacos",
    "takoyaki",
    "tiramisu",
    "tuna_tartare",
    "waffles"
]

# Mapping for common food names to class indices
FOOD_NAME_MAPPING = {
    "pizza": "pizza",
    "hamburger": "hamburger",
    "hot dog": "hot_dog",
    "ice cream": "ice_cream",
    "french fries": "french_fries",
    "sushi": "sushi",
    "tacos": "tacos",
    "pancakes": "pancakes",
    "waffles": "waffles",
    "pasta": "spaghetti_bolognese",
    "fried rice": "fried_rice",
    "salad": "caesar_salad",
    "sandwich": "club_sandwich",
    "cake": "chocolate_cake",
    "soup": "miso_soup",
    "chicken": "chicken_curry",
    "fish": "grilled_salmon",
    "rice": "fried_rice",
    "bread": "garlic_bread",
    "cheese": "grilled_cheese_sandwich"
}

def get_food_class_name(class_index):
    """Get human-readable food class name"""
    if 0 <= class_index < len(FOOD_CLASSES):
        food_name = FOOD_CLASSES[class_index]
        # Convert underscore to space and title case
        return food_name.replace('_', ' ').title()
    return "Unknown Food"

def search_food_class(query):
    """Search for food class by name"""
    query = query.lower().strip()
    
    # Direct mapping
    if query in FOOD_NAME_MAPPING:
        food_class = FOOD_NAME_MAPPING[query]
        if food_class in FOOD_CLASSES:
            return FOOD_CLASSES.index(food_class)
    
    # Partial match
    for i, food_name in enumerate(FOOD_CLASSES):
        if query in food_name.lower() or food_name.lower() in query:
            return i
    
    return -1  # Not found

# Additional nutritional categories for Indian foods
INDIAN_FOOD_EXTENSIONS = {
    "biryani": {"calories": 290, "protein": 8, "fat": 12, "carbs": 40},
    "dal": {"calories": 120, "protein": 9, "fat": 1, "carbs": 20},
    "roti": {"calories": 104, "protein": 4, "fat": 2, "carbs": 18},
    "naan": {"calories": 262, "protein": 8, "fat": 5, "carbs": 45},
    "curry": {"calories": 180, "protein": 12, "fat": 8, "carbs": 15},
    "samosa": {"calories": 308, "protein": 6, "fat": 13, "carbs": 43},
    "dosa": {"calories": 168, "protein": 4, "fat": 6, "carbs": 25},
    "idli": {"calories": 58, "protein": 2, "fat": 0.2, "carbs": 12}
}
