import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import io
import os
from models.food_classifier import FoodClassifier
from utils.image_processor import ImageProcessor
from utils.nutrition_api import NutritionAPI
from data.food_classes import FOOD_CLASSES

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

def main():
    st.set_page_config(
        page_title="NutriVision - Food Nutritional Analysis",
        page_icon="🍎",
        layout="wide"
    )
    
    # Header
    st.title("🍎 NutriVision")
    st.subheader("Automated Nutritional Analysis from Food Images")
    st.markdown("Upload a food image to get instant nutritional analysis using AI-powered image recognition.")
    
    # Load components
    try:
        model = load_model()
        image_processor, nutrition_api = load_processors()
    except Exception as e:
        st.error(f"Error loading application components: {str(e)}")
        st.stop()
    
    # Sidebar with information
    with st.sidebar:
        st.header("📊 About NutriVision")
        st.markdown("""
        **Features:**
        - AI-powered food recognition
        - Nutritional analysis (calories, protein, fat)
        - Support for 100+ food items
        - Real-time processing
        
        **Accuracy:**
        - Classification: 87.4%
        - Calorie prediction: ±23.6 kcal
        - Protein estimates: ±2.1g
        - Fat estimates: ±1.9g
        """)
        
        st.header("🎯 Best Results")
        st.markdown("""
        - Single food items
        - Clear, well-lit images
        - Fruits and rice dishes
        - Minimal occlusion
        """)
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📷 Upload Food Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a food image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of a single food item for best results"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Analyze button
            if st.button("🔍 Analyze Food", type="primary"):
                analyze_food(image, model, image_processor, nutrition_api, col2)
        else:
            st.info("Please upload a food image to begin analysis.")
    
    with col2:
        st.header("📋 Analysis Results")
        if uploaded_file is None:
            st.info("Upload an image to see nutritional analysis results here.")

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
                    st.markdown(f"**{predicted_food}**")
                with col_b:
                    st.metric("Confidence", f"{confidence_score:.1%}")
                
                # Get nutritional information
                with st.spinner("Fetching nutritional data..."):
                    nutrition_data = nutrition_api.get_nutrition_info(predicted_food)
                
                if nutrition_data:
                    display_nutrition_info(nutrition_data)
                else:
                    st.warning("⚠️ Nutritional data not available for this food item.")
                
                # Show top 3 predictions
                display_top_predictions(predictions, confidence_score)
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.info("Please try uploading a different image.")

def display_nutrition_info(nutrition_data):
    """Display nutritional information in a formatted way"""
    
    st.subheader("🥗 Nutritional Information")
    st.caption("Per 100g serving")
    
    # Main nutrition metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        calories = nutrition_data.get('calories', 0)
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
    
    # Accuracy disclaimer
    st.caption("⚠️ Nutritional values are estimates. Actual values may vary based on preparation method and portion size.")

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
