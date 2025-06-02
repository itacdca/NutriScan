import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import io
import os
import json
import cv2
from datetime import datetime, timedelta
from models.food_classifier import FoodClassifier
from utils.image_processor import ImageProcessor
from utils.nutrition_api import NutritionAPI
from data.food_classes import FOOD_CLASSES

# Initialize session state for user profile
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {
        'health_conditions': [],
        'allergies': [],
        'dietary_restrictions': [],
        'daily_goals': {'calories': 2000, 'protein': 50, 'fat': 65, 'carbs': 300},
        'food_history': [],
        'age': 30,
        'gender': 'Not specified',
        'weight': 70,
        'height': 170,
        'activity_level': 'moderate',
        'meal_preferences': [],
        'notification_settings': {
            'health_alerts': True,
            'meal_reminders': True,
            'progress_updates': True
        }
    }

# Initialize camera and voice states
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'voice_input' not in st.session_state:
    st.session_state.voice_input = ""
if 'meal_plan' not in st.session_state:
    st.session_state.meal_plan = []

# Health conditions and their dietary restrictions
HEALTH_CONDITIONS = {
    'diabetes': {
        'avoid': ['high sugar', 'refined carbs', 'white bread', 'candy'],
        'limit': ['carbohydrates', 'processed foods'],
        'recommendations': 'Monitor carbohydrate intake and choose complex carbs'
    },
    'hypertension': {
        'avoid': ['high sodium', 'processed meats', 'canned foods'],
        'limit': ['salt', 'sodium'],
        'recommendations': 'Limit sodium intake to less than 2300mg per day'
    },
    'heart_disease': {
        'avoid': ['trans fats', 'saturated fats', 'fried foods'],
        'limit': ['cholesterol', 'sodium'],
        'recommendations': 'Choose lean proteins and heart-healthy fats'
    },
    'kidney_disease': {
        'avoid': ['high potassium foods', 'excessive protein'],
        'limit': ['phosphorus', 'sodium', 'protein'],
        'recommendations': 'Monitor protein, phosphorus, and potassium intake'
    },
    'obesity': {
        'avoid': ['high calorie foods', 'sugary drinks', 'fast food'],
        'limit': ['calories', 'portion sizes'],
        'recommendations': 'Focus on portion control and nutrient-dense foods'
    }
}

# Initialize components
@st.cache_resource
def load_model():
    """Load the food classification model"""
    return FoodClassifier()

@st.cache_resource
def load_processors():
    """Load image processor and nutrition API"""
    image_processor = ImageProcessor()
    nutrition_api = NutritionAPI()
    return image_processor, nutrition_api

def setup_user_profile():
    """Setup comprehensive user health profile in sidebar"""
    with st.sidebar:
        st.header("👤 Complete Health Profile")
        
        # Personal Information
        with st.expander("📋 Personal Information", expanded=False):
            profile = st.session_state.user_profile
            
            col1, col2 = st.columns(2)
            with col1:
                profile['age'] = st.number_input("Age", min_value=13, max_value=120, value=profile['age'])
                profile['weight'] = st.number_input("Weight (kg)", min_value=30, max_value=300, value=profile['weight'])
            with col2:
                profile['gender'] = st.selectbox("Gender", ["Male", "Female", "Other", "Not specified"], 
                                                index=0 if profile['gender'] == "Male" else 1 if profile['gender'] == "Female" else 2)
                profile['height'] = st.number_input("Height (cm)", min_value=100, max_value=250, value=profile['height'])
            
            profile['activity_level'] = st.selectbox(
                "Activity Level",
                ["sedentary", "light", "moderate", "active", "very_active"],
                index=2,
                help="This affects your calorie needs"
            )
        
        # Health conditions
        st.subheader("🏥 Health Conditions")
        selected_conditions = st.multiselect(
            "Select your health conditions:",
            options=list(HEALTH_CONDITIONS.keys()),
            default=st.session_state.user_profile['health_conditions'],
            help="This helps us provide personalized dietary recommendations"
        )
        st.session_state.user_profile['health_conditions'] = selected_conditions
        
        # Allergies and restrictions
        st.subheader("🚫 Allergies & Restrictions")
        allergies = st.text_input(
            "Food allergies (comma-separated):",
            value=", ".join(st.session_state.user_profile['allergies']),
            help="e.g., nuts, dairy, gluten, shellfish"
        )
        if allergies:
            st.session_state.user_profile['allergies'] = [a.strip() for a in allergies.split(',')]
        
        # Dietary preferences
        dietary_prefs = st.multiselect(
            "Dietary Preferences:",
            ["vegetarian", "vegan", "keto", "paleo", "mediterranean", "low_carb", "low_fat"],
            default=st.session_state.user_profile.get('meal_preferences', [])
        )
        st.session_state.user_profile['meal_preferences'] = dietary_prefs
        
        # Dynamic daily goals calculation
        st.subheader("🎯 Daily Nutrition Goals")
        if st.button("📊 Calculate Personalized Goals"):
            calculate_personalized_goals()
        
        goals = st.session_state.user_profile['daily_goals']
        
        col1, col2 = st.columns(2)
        with col1:
            goals['calories'] = st.number_input("Calories", min_value=1000, max_value=5000, value=goals['calories'])
            goals['protein'] = st.number_input("Protein (g)", min_value=20, max_value=200, value=goals['protein'])
        with col2:
            goals['fat'] = st.number_input("Fat (g)", min_value=20, max_value=200, value=goals['fat'])
            goals['carbs'] = st.number_input("Carbs (g)", min_value=100, max_value=500, value=goals['carbs'])
        
        # Notification settings
        with st.expander("🔔 Notification Settings"):
            notif = st.session_state.user_profile['notification_settings']
            notif['health_alerts'] = st.checkbox("Health Alerts", value=notif['health_alerts'])
            notif['meal_reminders'] = st.checkbox("Meal Reminders", value=notif['meal_reminders'])
            notif['progress_updates'] = st.checkbox("Progress Updates", value=notif['progress_updates'])

def main():
    st.set_page_config(
        page_title="NutriVision - Smart Food Analysis",
        page_icon="🥗",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 0.5rem 0;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🥗 NutriVision</h1>
        <h3>Smart Food Analysis & Health Monitoring</h3>
        <p>AI-powered nutritional analysis tailored to your health needs</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Setup user profile
    setup_user_profile()
    
    # Load components
    try:
        model = load_model()
        image_processor, nutrition_api = load_processors()
    except Exception as e:
        st.error(f"Error loading application components: {str(e)}")
        st.stop()
    
    # Main tabs with enhanced features
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Food Analysis", "📊 Daily Summary", "🎯 Health Insights", "🍽️ Meal Planner", "📱 Smart Features"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("📷 Advanced Food Recognition")
            
            # Input method selection
            input_method = st.radio(
                "Choose input method:",
                ["📸 Upload Image", "📹 Camera Capture", "🎤 Voice Description"],
                horizontal=True
            )
            
            if input_method == "📸 Upload Image":
                uploaded_file = st.file_uploader(
                    "Choose a food image...",
                    type=['png', 'jpg', 'jpeg'],
                    help="Upload a clear image of a single food item for best results"
                )
                
                if uploaded_file is not None:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                    
                    # Enhanced analysis options
                    col_a, col_b = st.columns(2)
                    with col_a:
                        estimate_portion = st.checkbox("📏 Estimate Portion Size", help="Use object detection for portion estimation")
                    with col_b:
                        detect_ingredients = st.checkbox("🔍 Detect Ingredients", help="Identify individual ingredients")
                    
                    if st.button("🔍 Analyze Food", type="primary", use_container_width=True):
                        analyze_food_enhanced(image, model, image_processor, nutrition_api, col2, estimate_portion, detect_ingredients)
                        
            elif input_method == "📹 Camera Capture":
                st.info("📹 Camera capture feature - would access device camera in mobile app")
                
                # Simulated camera interface
                if st.button("📸 Take Photo"):
                    st.success("Photo captured! (Demo mode)")
                    # In real implementation, this would capture from camera
                    st.info("In a real mobile app, this would use the device camera for instant food recognition")
                    
                # Real-time features simulation
                st.subheader("🔄 Real-Time Features")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Detection Speed", "0.2s", "Real-time")
                with col_b:
                    st.metric("Confidence", "89%", "High")
                    
            elif input_method == "🎤 Voice Description":
                st.subheader("🎤 Voice Food Logging")
                voice_text = st.text_area(
                    "Describe your food (or use voice input):",
                    placeholder="e.g., 'I had a large chicken salad with dressing'",
                    help="Describe what you ate and the app will estimate nutrition"
                )
                
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("🎤 Start Voice Input"):
                        st.info("Voice recognition would be activated here")
                with col_b:
                    if st.button("📝 Process Description") and voice_text:
                        process_voice_input(voice_text, col2)
            
            # Quick actions
            st.subheader("⚡ Quick Actions")
            quick_cols = st.columns(3)
            with quick_cols[0]:
                if st.button("🍎 Common Foods"):
                    show_common_foods_modal()
            with quick_cols[1]:
                if st.button("📋 Scan Menu"):
                    st.info("OCR menu scanning - would read restaurant menus")
            with quick_cols[2]:
                if st.button("🏷️ Scan Barcode"):
                    st.info("Barcode scanning - would read product nutrition labels")
        
        with col2:
            st.header("📋 Enhanced Analysis Results")
            display_analysis_dashboard()
    
    with tab2:
        display_daily_summary()
    
    with tab3:
        display_health_insights()
    
    with tab4:
        display_meal_planner()
    
    with tab5:
        display_smart_features()

def analyze_food(image, model, image_processor, nutrition_api, result_column):
    """Analyze the uploaded food image and display results"""
    
    with result_column:
        # Show processing status
        with st.spinner("Processing image..."):
            try:
                # Preprocess image
                processed_image = image_processor.preprocess_image(image)
                
                # Classify food
                predictions, confidence = model.predict(processed_image)
                
                if confidence < 0.5:
                    st.warning("⚠️ Low confidence detection. Please try with a clearer image of a single food item.")
                    return
                
                # Get top prediction
                top_class_idx = np.argmax(predictions)
                predicted_food = FOOD_CLASSES[top_class_idx]
                confidence_score = float(confidence)
                
                # Display classification results
                st.success("✅ Food Classification Complete!")
                
                # Classification details
                st.subheader("🎯 Identified Food")
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    st.markdown(f"**{predicted_food.replace('_', ' ').title()}**")
                with col_b:
                    st.metric("Confidence", f"{confidence_score:.1%}")
                
                # Get nutritional information
                with st.spinner("Fetching nutritional data..."):
                    nutrition_data = nutrition_api.get_nutrition_info(predicted_food)
                
                if nutrition_data:
                    # Check for health warnings before displaying nutrition
                    health_warnings = check_health_compatibility(predicted_food, nutrition_data)
                    if health_warnings:
                        display_health_warnings(health_warnings)
                    
                    display_nutrition_info(nutrition_data, predicted_food)
                    
                    # Add to food history
                    add_to_food_history(predicted_food, nutrition_data)
                else:
                    st.warning("⚠️ Nutritional data not available for this food item.")
                
                # Show top 3 predictions
                display_top_predictions(predictions, confidence_score)
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.info("Please try uploading a different image.")

def check_health_compatibility(food_name, nutrition_data):
    """Check if food is compatible with user's health conditions"""
    warnings = []
    user_conditions = st.session_state.user_profile['health_conditions']
    
    for condition in user_conditions:
        if condition in HEALTH_CONDITIONS:
            condition_data = HEALTH_CONDITIONS[condition]
            
            # Check if food should be avoided
            for avoid_item in condition_data['avoid']:
                if avoid_item.lower() in food_name.lower():
                    warnings.append({
                        'type': 'danger',
                        'condition': condition,
                        'message': f"This food contains {avoid_item} which should be avoided with {condition.replace('_', ' ')}"
                    })
            
            # Check nutritional limits
            if condition == 'diabetes' and nutrition_data.get('carbs', 0) > 30:
                warnings.append({
                    'type': 'warning',
                    'condition': condition,
                    'message': f"High carbohydrate content ({nutrition_data['carbs']:.1f}g) - monitor blood sugar"
                })
            
            if condition == 'hypertension' and 'sodium' in food_name.lower():
                warnings.append({
                    'type': 'warning',
                    'condition': condition,
                    'message': "This food may be high in sodium - limit portion size"
                })
            
            if condition == 'obesity' and nutrition_data.get('calories', 0) > 300:
                warnings.append({
                    'type': 'warning',
                    'condition': condition,
                    'message': f"High calorie food ({nutrition_data['calories']:.0f} kcal) - consider portion control"
                })
    
    return warnings

def display_health_warnings(warnings):
    """Display health warnings based on user conditions"""
    for warning in warnings:
        if warning['type'] == 'danger':
            st.error(f"⚠️ **Health Alert**: {warning['message']}")
        else:
            st.warning(f"💡 **Health Tip**: {warning['message']}")

def add_to_food_history(food_name, nutrition_data):
    """Add analyzed food to user's daily history"""
    today = datetime.now().strftime("%Y-%m-%d")
    
    food_entry = {
        'date': today,
        'time': datetime.now().strftime("%H:%M"),
        'food': food_name.replace('_', ' ').title(),
        'calories': nutrition_data.get('calories', 0),
        'protein': nutrition_data.get('protein', 0),
        'fat': nutrition_data.get('fat', 0),
        'carbs': nutrition_data.get('carbs', 0)
    }
    
    st.session_state.user_profile['food_history'].append(food_entry)

def display_daily_summary():
    """Display daily nutrition summary"""
    st.header("📊 Today's Nutrition Summary")
    
    today = datetime.now().strftime("%Y-%m-%d")
    today_foods = [food for food in st.session_state.user_profile['food_history'] 
                   if food['date'] == today]
    
    if not today_foods:
        st.info("No food analyzed today. Start by analyzing some food images!")
        return
    
    # Calculate totals
    total_calories = sum(food['calories'] for food in today_foods)
    total_protein = sum(food['protein'] for food in today_foods)
    total_fat = sum(food['fat'] for food in today_foods)
    total_carbs = sum(food['carbs'] for food in today_foods)
    
    # Goals
    goals = st.session_state.user_profile['daily_goals']
    
    # Progress metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        progress = min(total_calories / goals['calories'], 1.0)
        st.metric("Calories", f"{total_calories:.0f}", f"Goal: {goals['calories']}")
        st.progress(progress)
    
    with col2:
        progress = min(total_protein / goals['protein'], 1.0)
        st.metric("Protein", f"{total_protein:.1f}g", f"Goal: {goals['protein']}g")
        st.progress(progress)
    
    with col3:
        progress = min(total_fat / goals['fat'], 1.0)
        st.metric("Fat", f"{total_fat:.1f}g", f"Goal: {goals['fat']}g")
        st.progress(progress)
    
    with col4:
        progress = min(total_carbs / goals['carbs'], 1.0)
        st.metric("Carbs", f"{total_carbs:.1f}g", f"Goal: {goals['carbs']}g")
        st.progress(progress)
    
    # Food log
    st.subheader("🍽️ Today's Food Log")
    if today_foods:
        df = pd.DataFrame(today_foods)
        st.dataframe(df[['time', 'food', 'calories', 'protein', 'fat', 'carbs']], use_container_width=True)

def display_health_insights():
    """Display personalized health insights"""
    st.header("🎯 Personalized Health Insights")
    
    user_conditions = st.session_state.user_profile['health_conditions']
    
    if not user_conditions:
        st.info("Set up your health profile to receive personalized recommendations.")
        return
    
    # Health condition specific advice
    for condition in user_conditions:
        if condition in HEALTH_CONDITIONS:
            condition_info = HEALTH_CONDITIONS[condition]
            
            with st.expander(f"📋 {condition.replace('_', ' ').title()} Management"):
                st.markdown(f"**Recommendation**: {condition_info['recommendations']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Foods to Avoid:**")
                    for item in condition_info['avoid']:
                        st.markdown(f"• {item}")
                
                with col2:
                    st.markdown("**Nutrients to Limit:**")
                    for item in condition_info['limit']:
                        st.markdown(f"• {item}")
    
    # Weekly progress (if history available)
    st.subheader("📈 Weekly Nutrition Trends")
    
    week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    week_foods = [food for food in st.session_state.user_profile['food_history'] 
                  if food['date'] >= week_ago]
    
    if week_foods:
        df = pd.DataFrame(week_foods)
        df['date'] = pd.to_datetime(df['date'])
        
        daily_calories = df.groupby('date')['calories'].sum()
        
        st.line_chart(daily_calories)
        st.caption("Daily calorie intake over the past week")
    else:
        st.info("Start using NutriVision daily to see your nutrition trends!")

def display_nutrition_info(nutrition_data, food_name):
    """Display nutritional information in a formatted way"""
    
    st.subheader("🥗 Nutritional Information")
    st.caption("Per 100g serving")
    
    # Main nutrition metrics with better styling
    col1, col2, col3 = st.columns(3)
    
    with col1:
        calories = nutrition_data.get('calories', 0)
        delta_color = "normal"
        if calories > 400:
            delta_color = "inverse"
        elif calories < 100:
            delta_color = "normal"
        
        st.metric(
            label="Calories",
            value=f"{calories:.0f}",
            help="Energy content in kcal"
        )
    
    with col2:
        protein = nutrition_data.get('protein', 0)
        st.metric(
            label="Protein",
            value=f"{protein:.1f}g",
            help="Protein content in grams"
        )
    
    with col3:
        fat = nutrition_data.get('fat', 0)
        st.metric(
            label="Fat",
            value=f"{fat:.1f}g",
            help="Fat content in grams"
        )
    
    # Additional nutrients if available
    if any(key in nutrition_data for key in ['carbs', 'fiber', 'sugar']):
        st.subheader("📊 Additional Nutrients")
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            if 'carbs' in nutrition_data:
                st.metric("Carbohydrates", f"{nutrition_data['carbs']:.1f}g")
        
        with col5:
            if 'fiber' in nutrition_data:
                st.metric("Fiber", f"{nutrition_data['fiber']:.1f}g")
        
        with col6:
            if 'sugar' in nutrition_data:
                st.metric("Sugar", f"{nutrition_data['sugar']:.1f}g")
    
    # Health score
    health_score = calculate_health_score(nutrition_data)
    st.subheader("🏆 Health Score")
    
    score_col1, score_col2 = st.columns([1, 3])
    with score_col1:
        st.metric("Score", f"{health_score}/100")
    with score_col2:
        st.progress(health_score / 100)
        if health_score >= 80:
            st.success("Excellent nutritional choice!")
        elif health_score >= 60:
            st.info("Good food choice with balanced nutrition")
        else:
            st.warning("Consider pairing with healthier options")
    
    # Portion size recommendation
    st.subheader("🥄 Recommended Portion")
    portion_advice = get_portion_advice(nutrition_data, food_name)
    st.info(portion_advice)
    
    # Accuracy disclaimer
    st.caption("⚠️ Nutritional values are estimates. Actual values may vary based on preparation method and portion size.")

def calculate_health_score(nutrition_data):
    """Calculate a health score based on nutritional content"""
    score = 50  # Base score
    
    # Positive factors
    protein = nutrition_data.get('protein', 0)
    fiber = nutrition_data.get('fiber', 0)
    
    if protein > 10:
        score += 20
    elif protein > 5:
        score += 10
    
    if fiber > 5:
        score += 15
    elif fiber > 2:
        score += 8
    
    # Negative factors
    calories = nutrition_data.get('calories', 0)
    fat = nutrition_data.get('fat', 0)
    sugar = nutrition_data.get('sugar', 0)
    
    if calories > 400:
        score -= 15
    if fat > 20:
        score -= 10
    if sugar > 15:
        score -= 10
    
    return max(0, min(100, score))

def get_portion_advice(nutrition_data, food_name):
    """Get portion size advice based on food type and nutrition"""
    calories = nutrition_data.get('calories', 0)
    
    if calories > 400:
        return "⚠️ High calorie food. Consider a smaller portion (50-75g) or pair with low-calorie vegetables."
    elif calories > 250:
        return "🥄 Moderate portion recommended (75-100g). Good as part of a balanced meal."
    else:
        return "✅ You can enjoy a regular portion (100-150g) of this nutritious food."

def calculate_personalized_goals():
    """Calculate personalized nutrition goals based on user profile"""
    profile = st.session_state.user_profile
    
    # Calculate BMR using Mifflin-St Jeor Equation
    if profile['gender'] == 'Male':
        bmr = 10 * profile['weight'] + 6.25 * profile['height'] - 5 * profile['age'] + 5
    else:
        bmr = 10 * profile['weight'] + 6.25 * profile['height'] - 5 * profile['age'] - 161
    
    # Activity multipliers
    activity_multipliers = {
        'sedentary': 1.2,
        'light': 1.375,
        'moderate': 1.55,
        'active': 1.725,
        'very_active': 1.9
    }
    
    # Calculate total daily energy expenditure
    tdee = bmr * activity_multipliers.get(profile['activity_level'], 1.55)
    
    # Set macronutrient goals
    calories = int(tdee)
    protein = int(profile['weight'] * 1.6)  # 1.6g per kg bodyweight
    fat = int(calories * 0.25 / 9)  # 25% of calories from fat
    carbs = int((calories - (protein * 4) - (fat * 9)) / 4)  # Remaining calories from carbs
    
    # Update goals
    goals = st.session_state.user_profile['daily_goals']
    goals['calories'] = calories
    goals['protein'] = protein
    goals['fat'] = fat
    goals['carbs'] = carbs
    
    st.success(f"Goals updated! Daily calories: {calories}, Protein: {protein}g, Fat: {fat}g, Carbs: {carbs}g")

def analyze_food_enhanced(image, model, image_processor, nutrition_api, result_column, estimate_portion, detect_ingredients):
    """Enhanced food analysis with portion estimation and ingredient detection"""
    
    with result_column:
        with st.spinner("Processing image with advanced features..."):
            try:
                # Standard analysis
                processed_image = image_processor.preprocess_image(image)
                predictions, confidence = model.predict(processed_image)
                
                if confidence < 0.5:
                    st.warning("⚠️ Low confidence detection. Please try with a clearer image of a single food item.")
                    return
                
                top_class_idx = np.argmax(predictions)
                predicted_food = FOOD_CLASSES[top_class_idx]
                confidence_score = float(confidence)
                
                st.success("✅ Enhanced Food Analysis Complete!")
                
                # Display results with enhancements
                st.subheader("🎯 Identified Food")
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    st.markdown(f"**{predicted_food.replace('_', ' ').title()}**")
                with col_b:
                    st.metric("Confidence", f"{confidence_score:.1%}")
                
                # Portion estimation
                if estimate_portion:
                    st.subheader("📏 Portion Analysis")
                    estimated_weight = estimate_food_portion(image, predicted_food)
                    st.info(f"Estimated portion size: {estimated_weight}g")
                    
                # Ingredient detection
                if detect_ingredients:
                    st.subheader("🔍 Ingredient Analysis")
                    ingredients = detect_food_ingredients(predicted_food)
                    for ingredient in ingredients:
                        st.write(f"• {ingredient}")
                
                # Get nutritional information
                with st.spinner("Fetching nutritional data..."):
                    nutrition_data = nutrition_api.get_nutrition_info(predicted_food)
                
                if nutrition_data:
                    # Check for health warnings
                    health_warnings = check_health_compatibility(predicted_food, nutrition_data)
                    if health_warnings:
                        display_health_warnings(health_warnings)
                    
                    # Allergy check
                    allergy_warnings = check_allergens(predicted_food)
                    if allergy_warnings:
                        st.error(f"🚨 **ALLERGY ALERT**: {allergy_warnings}")
                    
                    display_nutrition_info(nutrition_data, predicted_food)
                    add_to_food_history(predicted_food, nutrition_data)
                else:
                    st.warning("⚠️ Nutritional data not available for this food item.")
                
                display_top_predictions(predictions, confidence_score)
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")

def process_voice_input(voice_text, result_column):
    """Process voice input and extract nutrition information"""
    with result_column:
        st.subheader("🎤 Voice Input Analysis")
        st.write(f"Processing: '{voice_text}'")
        
        # Simple food extraction from text
        food_items = extract_foods_from_text(voice_text)
        
        if food_items:
            st.success(f"Detected foods: {', '.join(food_items)}")
            
            for food in food_items:
                # Load nutrition API
                nutrition_api = NutritionAPI()
                nutrition_data = nutrition_api.get_nutrition_info(food)
                
                if nutrition_data:
                    st.subheader(f"📊 {food.title()}")
                    display_nutrition_info(nutrition_data, food)
                    add_to_food_history(food, nutrition_data)
        else:
            st.warning("Could not identify specific foods from description. Please be more specific.")

def show_common_foods_modal():
    """Display common foods for quick logging"""
    st.subheader("🍎 Common Foods Quick Add")
    
    common_foods = [
        "apple", "banana", "orange", "chicken_breast", "rice", "bread", 
        "milk", "egg", "pasta", "potato", "tomato", "cheese"
    ]
    
    cols = st.columns(4)
    for i, food in enumerate(common_foods):
        with cols[i % 4]:
            if st.button(food.replace('_', ' ').title(), key=f"common_{food}"):
                nutrition_api = NutritionAPI()
                nutrition_data = nutrition_api.get_nutrition_info(food)
                if nutrition_data:
                    add_to_food_history(food, nutrition_data)
                    st.success(f"Added {food} to your food log!")

def display_analysis_dashboard():
    """Display enhanced analysis dashboard"""
    st.info("Upload an image or use voice input to see detailed nutritional analysis here.")
    
    # Show AI capabilities
    st.subheader("🤖 AI Capabilities")
    capabilities = [
        "🔍 Real-time food recognition",
        "📏 Portion size estimation", 
        "🧾 Ingredient identification",
        "⚠️ Health risk alerts",
        "🚨 Allergy detection",
        "📊 Nutritional scoring"
    ]
    
    for capability in capabilities:
        st.write(capability)

def display_meal_planner():
    """Display AI-powered meal planner"""
    st.header("🍽️ Smart Meal Planner")
    
    # Meal planning interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📅 Plan Your Meals")
        
        # Meal type selection
        meal_type = st.selectbox("Meal Type", ["Breakfast", "Lunch", "Dinner", "Snack"])
        
        # Dietary preferences consideration
        user_prefs = st.session_state.user_profile.get('meal_preferences', [])
        if user_prefs:
            st.info(f"Planning based on your preferences: {', '.join(user_prefs)}")
        
        # Generate meal suggestions
        if st.button("🎯 Generate Meal Suggestions"):
            suggestions = generate_meal_suggestions(meal_type, user_prefs)
            st.session_state.meal_suggestions = suggestions
        
        # Display suggestions
        if hasattr(st.session_state, 'meal_suggestions'):
            st.subheader("💡 Suggested Meals")
            for meal in st.session_state.meal_suggestions:
                with st.expander(f"🍽️ {meal['name']}"):
                    st.write(f"**Calories:** {meal['calories']}")
                    st.write(f"**Ingredients:** {', '.join(meal['ingredients'])}")
                    st.write(f"**Health Score:** {meal['health_score']}/100")
                    
                    if st.button(f"Add {meal['name']}", key=f"add_{meal['name']}"):
                        add_planned_meal_to_history(meal)
                        st.success(f"Added {meal['name']} to your food log!")
    
    with col2:
        st.subheader("📋 Weekly Meal Plan")
        display_weekly_meal_plan()
        
        # Grocery list generation
        if st.button("🛒 Generate Grocery List"):
            grocery_list = generate_grocery_list()
            st.subheader("🛍️ Your Grocery List")
            for category, items in grocery_list.items():
                st.write(f"**{category}:**")
                for item in items:
                    st.write(f"• {item}")

def display_smart_features():
    """Display smart features and integrations"""
    st.header("📱 Smart Features & Integration")
    
    # Feature categories
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔗 Health Integrations")
        
        # Simulated integrations
        integrations = [
            {"name": "Apple Health", "status": "Available", "icon": "🍎"},
            {"name": "Google Fit", "status": "Available", "icon": "🏃"},
            {"name": "MyFitnessPal", "status": "Coming Soon", "icon": "💪"},
            {"name": "Electronic Health Records", "status": "Available", "icon": "🏥"}
        ]
        
        for integration in integrations:
            col_a, col_b, col_c = st.columns([1, 2, 1])
            with col_a:
                st.write(integration["icon"])
            with col_b:
                st.write(integration["name"])
            with col_c:
                if integration["status"] == "Available":
                    st.success("✓")
                else:
                    st.info("⏳")
        
        if st.button("🔄 Sync Health Data"):
            st.success("Health data synchronized!")
    
    with col2:
        st.subheader("🤖 AI Assistant")
        
        # Chat interface
        st.write("Ask me anything about nutrition!")
        
        user_question = st.text_input("Your question:", placeholder="e.g., What foods are good for diabetes?")
        
        if st.button("💬 Ask AI") and user_question:
            ai_response = get_ai_nutrition_advice(user_question)
            st.info(f"🤖 NutriVision AI: {ai_response}")
        
        # Quick AI suggestions
        st.subheader("💡 Quick Tips")
        if st.session_state.user_profile['health_conditions']:
            for condition in st.session_state.user_profile['health_conditions']:
                tip = get_health_tip(condition)
                st.info(f"💡 {condition.title()}: {tip}")
    
    # Community features
    st.subheader("👥 Community Features")
    
    community_cols = st.columns(3)
    with community_cols[0]:
        st.metric("Community Challenges", "5 Active", "Join now!")
    with community_cols[1]:
        st.metric("Healthy Meals Shared", "1,234", "This week")
    with community_cols[2]:
        st.metric("Your Streak", "7 days", "Keep going!")
    
    if st.button("🏆 View Challenges"):
        display_community_challenges()

# Helper functions for enhanced features
def estimate_food_portion(image, food_name):
    """Estimate portion size from image"""
    # Simulate portion estimation
    base_portions = {
        'apple': 150, 'banana': 120, 'pizza': 200, 'hamburger': 250,
        'salad': 100, 'rice': 150, 'chicken': 150, 'pasta': 200
    }
    
    base_weight = base_portions.get(food_name.split('_')[0], 150)
    # Add some variation based on "image analysis"
    estimated = base_weight + np.random.randint(-30, 50)
    return max(50, estimated)

def detect_food_ingredients(food_name):
    """Detect ingredients in food"""
    ingredient_db = {
        'pizza': ['wheat flour', 'tomato sauce', 'mozzarella cheese', 'olive oil'],
        'hamburger': ['ground beef', 'wheat bun', 'lettuce', 'tomato', 'onion'],
        'salad': ['lettuce', 'tomato', 'cucumber', 'olive oil', 'vinegar'],
        'pasta': ['wheat flour', 'eggs', 'tomato sauce', 'cheese'],
        'chicken_curry': ['chicken breast', 'onion', 'garlic', 'curry spices', 'coconut milk']
    }
    
    return ingredient_db.get(food_name, ['main ingredient', 'seasoning', 'oil'])

def check_allergens(food_name):
    """Check for allergens based on user profile"""
    user_allergies = st.session_state.user_profile.get('allergies', [])
    
    allergen_db = {
        'pizza': ['gluten', 'dairy'],
        'pasta': ['gluten', 'eggs'],
        'hamburger': ['gluten', 'dairy'],
        'shrimp': ['shellfish'],
        'peanuts': ['nuts'],
        'milk': ['dairy']
    }
    
    food_allergens = allergen_db.get(food_name, [])
    
    for allergy in user_allergies:
        for allergen in food_allergens:
            if allergy.lower() in allergen.lower():
                return f"Contains {allergen} - you are allergic to {allergy}"
    
    return None

def extract_foods_from_text(text):
    """Extract food names from text description"""
    common_foods = ['chicken', 'beef', 'fish', 'rice', 'pasta', 'salad', 'apple', 'banana', 
                   'bread', 'cheese', 'milk', 'egg', 'potato', 'tomato', 'pizza', 'burger']
    
    found_foods = []
    text_lower = text.lower()
    
    for food in common_foods:
        if food in text_lower:
            found_foods.append(food)
    
    return found_foods

def generate_meal_suggestions(meal_type, preferences):
    """Generate AI meal suggestions based on preferences"""
    meal_db = {
        'Breakfast': [
            {'name': 'Avocado Toast', 'calories': 320, 'ingredients': ['whole grain bread', 'avocado', 'tomato'], 'health_score': 85},
            {'name': 'Greek Yogurt Bowl', 'calories': 250, 'ingredients': ['greek yogurt', 'berries', 'granola'], 'health_score': 90},
            {'name': 'Oatmeal with Fruits', 'calories': 280, 'ingredients': ['oats', 'banana', 'blueberries'], 'health_score': 88}
        ],
        'Lunch': [
            {'name': 'Quinoa Salad', 'calories': 420, 'ingredients': ['quinoa', 'vegetables', 'olive oil'], 'health_score': 92},
            {'name': 'Grilled Chicken Wrap', 'calories': 380, 'ingredients': ['chicken breast', 'whole wheat wrap', 'vegetables'], 'health_score': 78},
            {'name': 'Lentil Soup', 'calories': 300, 'ingredients': ['lentils', 'vegetables', 'herbs'], 'health_score': 85}
        ],
        'Dinner': [
            {'name': 'Salmon with Vegetables', 'calories': 450, 'ingredients': ['salmon', 'broccoli', 'sweet potato'], 'health_score': 95},
            {'name': 'Chicken Stir Fry', 'calories': 380, 'ingredients': ['chicken breast', 'mixed vegetables', 'brown rice'], 'health_score': 82},
            {'name': 'Vegetable Curry', 'calories': 350, 'ingredients': ['mixed vegetables', 'coconut milk', 'spices'], 'health_score': 88}
        ]
    }
    
    return meal_db.get(meal_type, [])

def display_weekly_meal_plan():
    """Display weekly meal planning interface"""
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    for day in days:
        with st.expander(f"📅 {day}"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("🌅 Breakfast: Oatmeal")
            with col2:
                st.write("🍽️ Lunch: Salad")
            with col3:
                st.write("🌙 Dinner: Salmon")

def generate_grocery_list():
    """Generate grocery list based on meal plan"""
    return {
        'Proteins': ['chicken breast', 'salmon', 'eggs', 'greek yogurt'],
        'Vegetables': ['broccoli', 'spinach', 'tomatoes', 'avocado'],
        'Grains': ['quinoa', 'oats', 'brown rice', 'whole grain bread'],
        'Fruits': ['bananas', 'blueberries', 'apples', 'oranges']
    }

def add_planned_meal_to_history(meal):
    """Add planned meal to food history"""
    today = datetime.now().strftime("%Y-%m-%d")
    
    food_entry = {
        'date': today,
        'time': datetime.now().strftime("%H:%M"),
        'food': meal['name'],
        'calories': meal['calories'],
        'protein': meal.get('protein', 20),
        'fat': meal.get('fat', 15),
        'carbs': meal.get('carbs', 30)
    }
    
    st.session_state.user_profile['food_history'].append(food_entry)

def get_ai_nutrition_advice(question):
    """Get AI nutrition advice based on question"""
    responses = {
        'diabetes': "Focus on complex carbohydrates, lean proteins, and high-fiber foods. Avoid sugary drinks and refined carbs.",
        'weight loss': "Create a caloric deficit with nutrient-dense foods. Include plenty of vegetables, lean proteins, and whole grains.",
        'protein': "Good protein sources include lean meats, fish, eggs, legumes, and dairy products.",
        'energy': "For sustained energy, choose complex carbs like oats, quinoa, and sweet potatoes.",
        'heart': "Heart-healthy foods include fatty fish, nuts, olive oil, and plenty of fruits and vegetables."
    }
    
    question_lower = question.lower()
    for key, response in responses.items():
        if key in question_lower:
            return response
    
    return "For personalized nutrition advice, consult with a registered dietitian. Focus on a balanced diet with variety."

def get_health_tip(condition):
    """Get health tips for specific conditions"""
    tips = {
        'diabetes': "Monitor carb portions and choose low glycemic foods",
        'hypertension': "Reduce sodium intake and increase potassium-rich foods",
        'heart_disease': "Include omega-3 fatty acids and limit saturated fats",
        'obesity': "Focus on portion control and increase physical activity"
    }
    
    return tips.get(condition, "Maintain a balanced diet and regular exercise")

def display_community_challenges():
    """Display community challenges"""
    challenges = [
        "🥗 Eat 5 servings of vegetables daily",
        "💧 Drink 8 glasses of water",
        "🏃 Walk 10,000 steps",
        "🍎 No processed foods for a week",
        "🥛 Include protein in every meal"
    ]
    
    st.subheader("🏆 Active Challenges")
    for challenge in challenges:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(challenge)
        with col2:
            if st.button("Join", key=f"join_{challenge}"):
                st.success("Joined!")

def display_top_predictions(predictions, top_confidence):
    """Display top 3 predictions with confidence scores"""
    
    st.subheader("🏆 Top Predictions")
    
    # Get top 3 predictions
    top_3_indices = np.argsort(predictions)[-3:][::-1]
    
    for i, idx in enumerate(top_3_indices):
        food_name = FOOD_CLASSES[idx]
        confidence = float(predictions[idx])
        
        # Create progress bar for confidence
        col_name, col_conf = st.columns([3, 1])
        
        with col_name:
            if i == 0:
                st.markdown(f"**1. {food_name}** ⭐")
            else:
                st.markdown(f"{i+1}. {food_name}")
        
        with col_conf:
            st.progress(confidence)
            st.caption(f"{confidence:.1%}")

if __name__ == "__main__":
    main()
