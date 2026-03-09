import streamlit as st
import requests
import os

# Use environment variable for deployment, fallback to local for development
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.title('Ai Chat Assistant')

st.markdown('Ask questions about the Dataset')
st.caption('⏳ The first question may take up to ~60s as the AI model initializes on the server.')

# Sidebar for API Key
with st.sidebar:
    st.header("⚙️ Configuration")
    user_api_key = st.text_input("Enter Groq API Key", type="password", help="Required if the backend doesn't have a default API key configured.")

# user input
query = st.text_input('Ask a question')

# button
if st.button('Ask'):
    if query.strip() == '':
        st.warning('Please enter a question')
    else:
        try:
            payload = {'query': query}
            if user_api_key.strip():
                payload['api_key'] = user_api_key.strip()
            
            with st.spinner('🤖 Thinking... (first query may take ~60s)'):
                response = requests.post(
                    f"{BACKEND_URL}/ask",
                    json=payload,
                    timeout=120
                )

            result = response.json()
            answer = result.get('answer', 'No answer found.')

            st.subheader('Ai Response')
            st.write(answer)

        except requests.exceptions.Timeout:
            st.error('⏱️ Request timed out. The server may be starting up — please try again in a minute.')
        except Exception as e:
            st.error(f"Error connecting to backend: {str(e)}")
            st.info(f"Backend URL: {BACKEND_URL}")