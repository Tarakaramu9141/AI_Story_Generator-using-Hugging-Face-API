import streamlit as st
import pandas as pd
import altair as alt
from src.sentiment import analyze_sentiment
from src.text_generation import generate_text
from huggingface_hub import InferenceClient

# Set page config
st.set_page_config(
    page_title="AI-Powered Story & Text Generator",
    page_icon=":robot:",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("AI-Powered  Story & Text Generator")
st.markdown("This app generates creative AI responses and analyzes their sentiment. Try different prompts and explore the results!")

# Parameter Conrol: Max Tokens
max_tokens = st.slider("Max Tokens", min_value=50, max_value=200, value=50, step=10, help="Controls the maximum length of the generated text")

with st.sidebar:
    st.title("About this app")
    st.write("""
    **Welcome to the AI-Powered Story & Text Generator!**  
    
**How it works:**
    - **Enter your query:** Use the text box to type a question or prompt.
    - **AI Response:** The app uses Gemma-2-27B to generate a creative response.
    - **Sentiment Analysis:** The response is analyzed for sentiment, which is displayed alongside.

    **Examples:**
    - "Tell me a story about a brave knight."
    - "What are the latest trends in AI?"
    - "Describe a futuristic city."
    """)

# Input field for Hugging Face API key (user must enter their own key)
api_key = st.text_input(
    "Enter your Hugging Face API key", 
    type="password")

# Restrict access if no API key is provided
if not api_key:
    st.warning("⚠️ Please enter your Hugging Face API key to continue.")
    st.stop()

# Initialize API client with user's key
client = InferenceClient(provider="hf-inference", api_key=api_key)


# Initialize conversation history in session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# User query input
user_query = st.text_input("Enter your query here")

if user_query:
    # Show a spinner while generating response
    with st.spinner("Generating response..."):
        # Generate response based on user query
        response = generate_text(user_query, client, max_tokens=max_tokens)
        # Analyze sentiment of the response
        sentiment, score = analyze_sentiment(response)

    # Save the conversation to session state
    st.session_state.conversation.append({
        "query": user_query,
        "response": response,
        "sentiment": sentiment,
        "score": score
    })
    
    # Display the latest response and sentiment analysis in two columns
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("AI Response:")
        st.write(response)

    with col2:
        sentiment, score = analyze_sentiment(response)
        st.subheader("Sentiment Analysis:")
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Score: {score:.3f}")

    # Convert sentiment to numeric value for visualization
    # (Positive -> +score, Negative -> -score)
    if sentiment == "POSITIVE":
        numeric_score = score
    else:
        numeric_score = -score

    # Visualize sentiment score using a bar chart 
    sentiment_data = pd.DataFrame({
        "Sentiment": [sentiment], 
        "Score": [numeric_score]
    })

    # Create a bar chart to visualize sentiment score
    chart = alt.Chart(sentiment_data).mark_bar().encode(
        x=alt.X('Score:Q', 
                scale=alt.Scale(domain=[-1, 1]),
                title='Sentiment Score'),
        color=alt.condition(
            alt.datum.Score < 0,
            alt.value("red"),  # Negative
            alt.value("green") # Positive
        )
    ).properties(
        width=300,
        height=100
    )
    st.altair_chart(chart, use_container_width=True)
    
    # Expandable section to show the conversation history
    with st.expander("Show Full Response"):
        for idx, convo in enumerate(st.session_state.conversation, start=1):
            st.write(f"**Query {idx}:** {convo['query']}")
            st.write(f"**AI Response {idx}:** {convo['response']}")
            st.write(f"**Sentiment: {convo['sentiment']} (Score: {convo['score']})**")
            st.write("---")







