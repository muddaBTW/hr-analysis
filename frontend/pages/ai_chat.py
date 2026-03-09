import streamlit as st
import requests

st.title('Ai Chat Assistant')

st.markdown('Ask questions about the Dataset')

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
                
            response = requests.post(
                "http://127.0.0.1:8000/ask",
                json=payload
            )

            result = response.json()
            answer = result.get('answer', 'No answer found.')

            st.subheader('Ai Response')
            st.write(answer)

        except:
            st.error('Backend not running')