import requests
import streamlit as st
import os
import json

# Load API Key
API_KEY = os.getenv("HUGGINGFACE_API_KEY")
if not API_KEY:
    st.error("Hugging Face API key is missing. Set it as an environment variable.")
    st.stop()

st.write("API Key Loaded Successfully!")

# List of available models for user selection with smaller, more reliable models
model_list = [
    "openai-community/gpt2",           # Small OpenAI GPT-2 model
    "facebook/opt-125m",               # Small OPT model from Meta
    "mistralai/Mistral-7B-Instruct-v0.1",          
]

# Set headers for Hugging Face API requests
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# Function to call Hugging Face models with improved response handling
def query_model(model_name, prompt):
    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
    
    # Increase token limit for more complete responses
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 250,  # Increased from 50 to get more complete responses
            "temperature": 0.7,
            "return_full_text": False  # For some models, this prevents returning the prompt
        }
    }
    
    # Special handling for instruction-tuned models
    if "Instruct" in model_name or "instruct" in model_name:
        # Format prompt appropriately for instruction models
        if not prompt.startswith("<s>") and not prompt.startswith("[INST]"):
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            payload["inputs"] = formatted_prompt
    
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        
        # Handle different response formats
        if isinstance(result, list):
            if len(result) > 0:
                # Some models return a list with a dictionary containing 'generated_text'
                if isinstance(result[0], dict) and 'generated_text' in result[0]:
                    return result[0]['generated_text']
                # Others might return a list of strings or other formats
                return str(result[0])
        elif isinstance(result, dict):
            # Some models return a dictionary with 'generated_text'
            if 'generated_text' in result:
                return result['generated_text']
            # Or a dictionary with other keys
            return str(result)
        
        # If we can't parse it specifically, return the raw result as string
        return str(result)
    else:
        return f"Error {response.status_code}: {response.text}"

# Function to compare outputs using a simplified approach
def compare_outputs(model1_output, model2_output):
    # Simple heuristic for comparing outputs
    # This avoids API calls to another model which might timeout
    
    # Check for length (longer responses often have more content)
    len1 = len(model1_output) if isinstance(model1_output, str) else 0
    len2 = len(model2_output) if isinstance(model2_output, str) else 0
    
    # Check for coherence (fewer error messages or JSON artifacts)
    errors1 = "error" in model1_output.lower() if isinstance(model1_output, str) else True
    errors2 = "error" in model2_output.lower() if isinstance(model2_output, str) else True
    
    # Calculate a basic score (this could be improved with more sophisticated metrics)
    score1 = len1 * (0.5 if errors1 else 1.0)
    score2 = len2 * (0.5 if errors2 else 1.0)
    
    if score1 > score2:
        return {"better_model": "Model 1", "probability": min(0.9, max(0.6, score1/(score1+score2)))}
    elif score2 > score1:
        return {"better_model": "Model 2", "probability": min(0.9, max(0.6, score2/(score1+score2)))}
    else:
        return {"better_model": "Unknown", "probability": 0.5}

# Streamlit UI
st.title("AI Model Comparison Tool")

# User selects two models
model1 = st.selectbox("Select first model", model_list)
model2 = st.selectbox("Select second model", model_list, index=1)  # Default to second model

# User enters prompt
user_prompt = st.text_area("Enter your prompt:", height=150)

if st.button("Compare Models") and model1 and model2 and user_prompt:
    # Show progress
    with st.spinner(f"Calling {model1}..."):
        output1 = query_model(model1, user_prompt)
    
    with st.spinner(f"Calling {model2}..."):
        output2 = query_model(model2, user_prompt)

    # Display results from selected models
    st.subheader(f"Output from {model1}:")
    st.write(output1)
    
    # Add a download button for the first model's output
    st.download_button(
        label=f"Download {model1} output",
        data=output1,
        file_name=f"{model1.replace('/', '_')}_output.txt",
        mime="text/plain"
    )

    st.subheader(f"Output from {model2}:")
    st.write(output2)
    
    # Add a download button for the second model's output
    st.download_button(
        label=f"Download {model2} output",
        data=output2,
        file_name=f"{model2.replace('/', '_')}_output.txt",
        mime="text/plain"
    )

    # Compare using simplified approach
    with st.spinner("Comparing outputs..."):
        comparison_result = compare_outputs(output1, output2)

    # Display comparison result
    st.subheader("Comparison Result:")
    better_model = comparison_result.get('better_model', 'Unknown')
    probability = comparison_result.get('probability', 0.5)
    
    if better_model == "Model 1":
        winner = model1
    elif better_model == "Model 2":
        winner = model2
    else:
        winner = "Unable to determine"

    st.write(f"The analysis predicts that {winner} is better with a probability of {probability:.2f}")

    # Optional user feedback
    user_choice = st.radio("Which model do you think gave a better response?", (model1, model2))
      
    # Display technical details in an expandable section
    with st.expander("Technical Details"):
        st.write("Model 1 output length:", len(output1) if isinstance(output1, str) else 0)
        st.write("Model 2 output length:", len(output2) if isinstance(output2, str) else 0)
        st.write("Comparison method: Length and error detection heuristic")