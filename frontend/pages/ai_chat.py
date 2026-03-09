import streamlit as st
import requests

st.title('Ai Chat Assistant')

st.markdown('Ask questions about the Dataset')

# user input
query = st.text_input('Ask a question')

# button
if st.button('Ask'):
    if query.strip() == '':
        st.warning('Please enter a question')
    else:
        try:
            response = requests.post(
                "http://127.0.0.1:8000/ask",
                json={'query':query}
            )

            result = response.json()
            answer = result['answer']

            st.subheader('Ai Response')
            st.write(answer)

        except:
            st.error('Backend not running')