import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
news_text = "America launches Chandrayaan-3 to the moon"

response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    },
    json={
        "model": "anthropic/claude-3-haiku",  # ✅ Valid model
        "messages": [
            {
                "role": "system",
                "content": "You are a fact-checking assistant. If the news sounds fake, return a possible correction in 1-2 sentences."
            },
            {
                "role": "user",
                "content": news_text
            }
        ]
    }
)

print(response.status_code)
print(response.json())

# The code which i used in each state rep in try:


# import streamlit as st
# import re
# import joblib
# import numpy as np
# from bs4 import BeautifulSoup

# # Load model and components
# model = joblib.load("fake_news_model.pkl")
# vectorizer = joblib.load("tfidf_vectorizer.pkl")
# threshold = joblib.load("optimal_threshold.pkl")

# # --- FAQ Responses ---
# custom_responses = {
#     "how": "🧠 I analyze the text using TF-IDF, then classify it as real or fake using a trained logistic regression model.",
#     "model": "🔍 I'm powered by a Logistic Regression classifier — it's interpretable and accurate for text data.",
#     "data": "📚 The training data includes real news from trusted sources and fake news from known misinformation sources.",
#     "tfidf": "📊 TF-IDF stands for Term Frequency-Inverse Document Frequency. It helps identify informative words in a document.",
#     "logistic regression": "📈 Logistic Regression is a classification algorithm that predicts the probability of a label — in this case, real or fake.",
#     "accuracy": "✅ The model achieves around 92% accuracy on the test set. Confidence scores provide more nuanced predictions.",
#     "trust": "🤔 Use me as a first filter. Always verify critical information with trusted sources.",
#     "confidence": "📏 Confidence reflects how sure the model is about its prediction — closer to 1.0 means higher certainty.",
#     "sarcasm": "😅 Sarcasm and satire are tricky for models — I might misclassify such content.",
#     "tech stack": "💻 Python, scikit-learn, Streamlit, and joblib for saving and loading model components.",
#     "goal": "🎯 To demonstrate how machine learning and NLP can help in identifying misinformation.",
#     "who": "👨‍💻 This project was built by a student to explore NLP's role in combating fake news.",
#     "test": "📰 Yes, paste any article or headline and I'll try to predict its authenticity.",
#     "languages": "🌍 Currently only English is supported. More languages may be added in future.",
#     "contribute": "🤝 Contributions are welcome! Suggest improvements, add data, or extend the app!",
#     "ai": "🤖 Yes — I'm a basic AI system trained using supervised machine learning.",
#     "fake news": "🚨 Fake news is deliberately misleading or false information spread via media.",
#     "machine learning": "🧠 Machine learning helps systems learn from data. I was trained on labeled news articles.",
#     "hello": "👋 Hello! I'm here to help you detect fake news.",
#     "hi": "🙋‍♂️ Hi there! Need help with fake news detection?",
#     "hey": "Hey! 👋 Paste a headline or ask about the model.",
#     "bye": "👋 Goodbye! Stay critical, stay informed.",
#     "thank you": "🙏 You're welcome! Let me know if you have more questions.",
#     "thanks": "😊 No problem! Glad to help."
# }

# # --- Helper Functions ---
# def get_custom_response(text):
#     text_lower = text.lower().strip()
#     matched_keywords = []
#     for keyword, response in custom_responses.items():
#         if keyword in text_lower or any(word in text_lower for word in keyword.split()):
#             matched_keywords.append((keyword, response))

#     if matched_keywords:
#         matched_keywords.sort(key=lambda x: -len(x[0]))
#         return matched_keywords[0][1]

#     greetings = ["hi", "hii", "hello", "hey", "heyy", "yo", "hola", "namaste"]
#     if any(greet in text_lower for greet in greetings):
#         return "👋 Hello! I'm here to help you detect fake news or answer questions about the model."

#     return None

# def extract_title_content(text):
#     lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
#     meaningful = [l for l in lines if not re.search(r'\b(fake|real|check|news|bot|tell me|is this)\b', l, re.IGNORECASE)]
#     title = meaningful[0] if meaningful else ""
#     content = " ".join(meaningful[1:]) if len(meaningful) > 1 else ""
#     return title, content

# def clean_html_to_text(html):
#     try:
#         soup = BeautifulSoup(html, "html.parser")
#         return soup.get_text(separator="\n").strip()
#     except:
#         return html

# def generate_chat_log():
#     log = ""
#     for msg in st.session_state.get("messages", []):
#         role = "You" if msg["role"] == "user" else "Bot"
#         content = msg["content"]
#         if role == "Bot":
#             content = clean_html_to_text(content)
#         log += f"{role}: {content}\n\n"
#     return log

# # --- Sidebar ---
# with st.sidebar:
#     st.header("📁 Downloads")
#     st.download_button("📥 Download Chat Log", generate_chat_log(), "chat_history.txt")
#     if st.session_state.get("messages"):
#         last_msg = st.session_state.messages[-1]
       
# # --- Main Interface ---
# st.title("📰 Fake News Detector Chatbot")
# st.markdown("""
#     <div style='background-color:red; padding: 10px; border-left: 5px solid #ffa500; border-radius: 5px;'>
#         ⚠️ <strong>Disclaimer:</strong> This tool is for educational use only. Predictions may not always be accurate. Please verify news independently.
#     </div>
# """, unsafe_allow_html=True)

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# if not st.session_state.messages:
#     st.chat_message("assistant").markdown("👋 Hi! I'm here to help detect fake news. Paste a news article or ask about the model!")

# user_input = st.chat_input("Paste your news article or ask a question...")
# if user_input:
#     st.session_state.messages.append({"role": "user", "content": user_input})

#     title, content = extract_title_content(user_input)
#     combined = f"{title} {content}".strip()

#     if len(combined.split()) < 6:
#         bot_response = get_custom_response(user_input) or "❓ I'm not sure how to respond to that. Try rephrasing or provide a fuller article."
#     else:
#         vec = vectorizer.transform([combined])
#         prob = model.predict_proba(vec)[0][1]
#         prediction = "FAKE" if prob >= threshold else "REAL"
#         confidence = round(prob if prediction == "FAKE" else 1 - prob, 2)

#         if prediction == "REAL":
#             bot_response = f"""
#             <div style='background-color:#d4edda; padding: 10px; border-left: 5px solid #28a745; border-radius: 5px;'>
#                 📢 <strong style='color:#155724;'>This news appears to be real</strong><br><br>
#                 <span style='color:#1b2e1f;'>🔍 <strong>Confidence:</strong> {round(confidence * 100, 2)}%</span>
#             </div>
#             """
#         else:
#             bot_response = f"""
#             <div style='background-color:#f8d7da; padding: 10px; border-left: 5px solid #dc3545; border-radius: 5px;'>
#                 🚨 <strong style='color:#721c24;'>This news appears to be fake</strong><br><br>
#                 <span style='color:#4f1216;'>🔍 <strong>Confidence:</strong> {round(confidence * 100, 2)}%</span>
#             </div>
#             """

#     st.session_state.messages.append({"role": "assistant", "content": bot_response})

# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         if msg["role"] == "assistant" and ("appears to be" in msg["content"]):
#             st.markdown(msg["content"], unsafe_allow_html=True)
#         else:
#             st.markdown(msg["content"])


#try 2
# main.py
# import streamlit as st
# import re
# import joblib
# import numpy as np
# from bs4 import BeautifulSoup
# import os
# import time
# import requests
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()
# API_KEY = os.getenv("OPENROUTER_API_KEY")

# # Load ML model and vectorizer
# model = joblib.load("fake_news_model.pkl")
# vectorizer = joblib.load("tfidf_vectorizer.pkl")
# threshold = joblib.load("optimal_threshold.pkl")

# # Claude API Function
# def fetch_claude_fact_check(user_text):
#     url = "https://openrouter.ai/api/v1/chat/completions"
#     headers = {
#         "Authorization": f"Bearer {API_KEY}",
#         "Content-Type": "application/json"
#     }
#     data = {
#         "model": "anthropic/claude-3-haiku",
#         "messages": [
#             {"role": "system", "content": "You are a helpful fact-checking assistant. When a news statement is likely fake, provide what might be true instead."},
#             {"role": "user", "content": f"Fact-check this news: {user_text}. What might be true instead?"}
#         ]
#     }
#     try:
#         response = requests.post(url, headers=headers, json=data, timeout=10)
#         if response.status_code == 200:
#             return response.json()['choices'][0]['message']['content']
#         else:
#             return "⚠️ Claude API failed to respond."
#     except Exception as e:
#         return f"⚠️ Claude API error: {str(e)}"

# # Response mapping
# def get_custom_response(text):
#     text = text.lower().strip()
#     for keyword, response in custom_responses.items():
#         if keyword in text:
#             return response
#     greetings = ["hi", "hello", "hey", "yo", "hola"]
#     if any(greet in text for greet in greetings):
#         return "👋 Hello! I'm here to help you detect fake news or answer questions about the model."
#     return None

# # Preprocess input
# def extract_title_content(text):
#     lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
#     meaningful = [l for l in lines if not re.search(r'\b(fake|real|check|news|bot|tell me|is this)\b', l, re.IGNORECASE)]
#     title = meaningful[0] if meaningful else ""
#     content = " ".join(meaningful[1:]) if len(meaningful) > 1 else ""
#     return title, content

# def clean_html_to_text(html):
#     try:
#         return BeautifulSoup(html, "html.parser").get_text(separator="\n").strip()
#     except:
#         return html

# # Session and UI setup
# st.set_page_config(page_title="Fake News Detector", page_icon="🧠")
# st.title("📰 Fake News Detector Chatbot")
# st.markdown("""
#     <div style='background-color:red; padding: 10px; border-left: 5px solid #ffa500; border-radius: 5px;'>
#         ⚠️ <strong>Disclaimer:</strong> This tool is for educational use only. Always verify important news independently.
#     </div>
# """, unsafe_allow_html=True)

# # Sidebar downloads
# with st.sidebar:
#     st.header("📁 Downloads")
#     def generate_chat_log():
#         return "\n\n".join([f"{'You' if msg['role']=='user' else 'Bot'}: {msg['content']}" for msg in st.session_state.get("messages", [])])
#     st.download_button("📥 Download Chat Log", generate_chat_log(), "chat_history.txt")

# # State messages
# if "messages" not in st.session_state:
#     st.session_state.messages = []
# if not st.session_state.messages:
#     st.chat_message("assistant").markdown("👋 Hi! I'm here to help detect fake news. Paste a news article or ask about the model!")

# # User input
# user_input = st.chat_input("Paste your news article or ask a question...")
# if user_input:
#     st.session_state.messages.append({"role": "user", "content": user_input})
#     title, content = extract_title_content(user_input)
#     combined = f"{title} {content}".strip()

#     # Classification
#     if len(combined.split()) < 6:
#         response = get_custom_response(user_input) or "❓ I'm not sure how to respond to that. Try rephrasing or provide a fuller article."
#         st.session_state.messages.append({"role": "assistant", "content": response})
#     else:
#         vec = vectorizer.transform([combined])
#         prob = model.predict_proba(vec)[0][1]
#         prediction = "FAKE" if prob >= threshold else "REAL"
#         confidence = round(prob if prediction == "FAKE" else 1 - prob, 2)

#         if prediction == "REAL":
#             html_response = f"""
#                 <div style='background-color:#d4edda; padding: 10px; border-left: 5px solid #28a745; border-radius: 5px;'>
#                     📢 <strong style='color:#155724;'>This news appears to be real</strong><br><br>
#                     <span style='color:#1b2e1f;'>🔍 <strong>Confidence:</strong> {round(confidence * 100, 2)}%</span>
#                 </div>
#             """
#         else:
#             with st.spinner("🧠 Thinking about what might be true instead..."):
#                 time.sleep(2)  # Simulate loading
#                 fact_correction = fetch_claude_fact_check(user_input)
#             html_response = f"""
#                 <div style='background-color:#f8d7da; padding: 10px; border-left: 5px solid #dc3545; border-radius: 5px;'>
#                     🚨 <strong style='color:#721c24;'>This news appears to be fake</strong><br><br>
#                     <span style='color:#4f1216;'>🔍 <strong>Confidence:</strong> {round(confidence * 100, 2)}%</span><br><br>
#                     <div style='color:#721c24;'>📰 <strong>What might be true:</strong><br>{fact_correction}</div>
#                 </div>
#             """

#         st.session_state.messages.append({"role": "assistant", "content": html_response})

# # Display messages
# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         if msg["role"] == "assistant" and ("appears to be" in msg["content"] or "<div" in msg["content"]):
#             st.markdown(msg["content"], unsafe_allow_html=True)
#         else:
#             st.markdown(msg["content"])

# # FAQ responses
# theme_emoji = "🧠"
# custom_responses = {
#     "how": f"{theme_emoji} I analyze the text using TF-IDF and classify it as real or fake using a Logistic Regression model.",
#     "model": "🔍 I'm powered by Logistic Regression — it's interpretable and effective for text classification.",
#     "data": "📚 The model was trained on a dataset of real and fake news articles from known sources.",
#     "tfidf": "📊 TF-IDF stands for Term Frequency-Inverse Document Frequency. It helps highlight important words.",
#     "logistic regression": "📈 Logistic Regression is a statistical model used for binary classification.",
#     "accuracy": "✅ The model achieves ~92% accuracy on test data.",
#     "trust": "🤔 Always double-check news from multiple trusted sources.",
#     "confidence": "📏 Confidence indicates how sure the model is about its prediction.",
#     "sarcasm": "😅 Sarcasm and satire can fool even advanced models.",
#     "tech stack": "💻 Python, scikit-learn, Streamlit, joblib, and Claude via OpenRouter API.",
#     "goal": "🎯 To demonstrate how machine learning can help fight misinformation.",
#     "fake news": "🚨 Fake news is false info spread deliberately to mislead.",
#     "machine learning": "🧠 ML lets systems learn from data to make predictions.",
#     "languages": "🌍 Currently, only English is supported.",
#     "ai": "🤖 I'm an AI assistant trained using NLP techniques.",
#     "thank you": "🙏 You're welcome! Stay informed.",
#     "thanks": "😊 No problem! Happy to help."
# }


# try 3 (for copy option of what might be true)

# import streamlit as st
# import re
# import joblib
# import numpy as np
# from bs4 import BeautifulSoup
# import os
# import time
# import requests
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()
# API_KEY = os.getenv("OPENROUTER_API_KEY")

# # Load ML model and vectorizer
# model = joblib.load("fake_news_model.pkl")
# vectorizer = joblib.load("tfidf_vectorizer.pkl")
# threshold = joblib.load("optimal_threshold.pkl")

# # Claude API Function
# def fetch_claude_fact_check(user_text):
#     url = "https://openrouter.ai/api/v1/chat/completions"
#     headers = {
#         "Authorization": f"Bearer {API_KEY}",
#         "Content-Type": "application/json"
#     }
#     data = {
#         "model": "anthropic/claude-3-haiku",
#         "messages": [
#             {"role": "system", "content": "You are a helpful fact-checking assistant. When a news statement is likely fake, provide what might be true instead."},
#             {"role": "user", "content": f"Fact-check this news: {user_text}. What might be true instead?"}
#         ]
#     }
#     try:
#         response = requests.post(url, headers=headers, json=data, timeout=10)
#         if response.status_code == 200:
#             return response.json()['choices'][0]['message']['content']
#         else:
#             return "⚠️ Claude API failed to respond."
#     except Exception as e:
#         return f"⚠️ Claude API error: {str(e)}"

# # Response mapping
# def get_custom_response(text):
#     text = text.lower().strip()
#     for keyword, response in custom_responses.items():
#         if keyword in text:
#             return response
#     greetings = ["hi", "hello", "hey", "yo", "hola"]
#     if any(greet in text for greet in greetings):
#         return "👋 Hello! I'm here to help you detect fake news or answer questions about the model."
#     return None

# # Preprocess input
# def extract_title_content(text):
#     lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
#     meaningful = [l for l in lines if not re.search(r'\b(fake|real|check|news|bot|tell me|is this)\b', l, re.IGNORECASE)]
#     title = meaningful[0] if meaningful else ""
#     content = " ".join(meaningful[1:]) if len(meaningful) > 1 else ""
#     return title, content

# def clean_html_to_text(html):
#     try:
#         return BeautifulSoup(html, "html.parser").get_text(separator="\n").strip()
#     except:
#         return html

# # Session and UI setup
# st.set_page_config(page_title="Fake News Detector", page_icon="🧠")
# st.title("📰 Fake News Detector Chatbot")
# st.markdown("""
#     <div style='background-color:red; padding: 10px; border-left: 5px solid #ffa500; border-radius: 5px;'>
#         ⚠️ <strong>Disclaimer:</strong> This tool is for educational use only. Always verify important news independently.
#     </div>
# """, unsafe_allow_html=True)

# # Sidebar downloads
# with st.sidebar:
#     st.header("📁 Downloads")
#     def generate_chat_log():
#         return "\n\n".join([f"{'You' if msg['role']=='user' else 'Bot'}: {msg['content']}" for msg in st.session_state.get("messages", [])])
#     st.download_button("📥 Download Chat Log", generate_chat_log(), "chat_history.txt")

# # State messages
# if "messages" not in st.session_state:
#     st.session_state.messages = []
# if not st.session_state.messages:
#     st.chat_message("assistant").markdown("👋 Hi! I'm here to help detect fake news. Paste a news article or ask about the model!")

# # User input
# user_input = st.chat_input("Paste your news article or ask a question...")
# if user_input:
#     st.session_state.messages.append({"role": "user", "content": user_input})
#     title, content = extract_title_content(user_input)
#     combined = f"{title} {content}".strip()

#     if len(combined.split()) < 6:
#         response = get_custom_response(user_input) or "❓ I'm not sure how to respond to that. Try rephrasing or provide a fuller article."
#         st.session_state.messages.append({"role": "assistant", "content": response})
#     else:
#         vec = vectorizer.transform([combined])
#         prob = model.predict_proba(vec)[0][1]
#         prediction = "FAKE" if prob >= threshold else "REAL"
#         confidence = round(prob if prediction == "FAKE" else 1 - prob, 2)

#         if prediction == "REAL":
#             html_response = f"""
#                 <div style='background-color:#d4edda; padding: 10px; border-left: 5px solid #28a745; border-radius: 5px;'>
#                     📢 <strong style='color:#155724;'>This news appears to be real</strong><br><br>
#                     <span style='color:#1b2e1f;'>🔍 <strong>Confidence:</strong> {round(confidence * 100, 2)}%</span>
#                 </div>
#             """
#         else:
#             spinner_placeholder = st.empty()
#             with spinner_placeholder.container():
#                 with st.spinner("🧠 Thinking about what might be true instead..."):
#                     time.sleep(2)
#                     fact_correction = fetch_claude_fact_check(user_input)
#             spinner_placeholder.empty()

#             html_response = f"""
#                 <div style='background-color:#f8d7da; padding: 10px; border-left: 5px solid #dc3545; border-radius: 5px;'>
#                     🚨 <strong style='color:#721c24;'>This news appears to be fake</strong><br><br>
#                     <span style='color:#4f1216;'>🔍 <strong>Confidence:</strong> {round(confidence * 100, 2)}%</span><br><br>
#                     <div style='background-color:#1e1e1e; color:#f1f1f1; padding:10px; border-radius:10px; font-family:monospace;'>
#                         <strong>What might be true:</strong><br>
#                         {fact_correction}
#                     </div>
#                 </div>
#             """

#         st.session_state.messages.append({"role": "assistant", "content": html_response})

# # Display messages
# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         if msg["role"] == "assistant" and ("appears to be" in msg["content"] or "<div" in msg["content"]):
#             st.markdown(msg["content"], unsafe_allow_html=True)
#         else:
#             st.markdown(msg["content"])

# # FAQ responses
# theme_emoji = "🧠"
# custom_responses = {
#     "how": f"{theme_emoji} I analyze the text using TF-IDF and classify it as real or fake using a Logistic Regression model.",
#     "model": "🔍 I'm powered by Logistic Regression — it's interpretable and effective for text classification.",
#     "data": "📚 The model was trained on a dataset of real and fake news articles from known sources.",
#     "tfidf": "📊 TF-IDF stands for Term Frequency-Inverse Document Frequency. It helps highlight important words.",
#     "logistic regression": "📈 Logistic Regression is a statistical model used for binary classification.",
#     "accuracy": "✅ The model achieves ~92% accuracy on test data.",
#     "trust": "🤔 Always double-check news from multiple trusted sources.",
#     "confidence": "📏 Confidence indicates how sure the model is about its prediction.",
#     "sarcasm": "😅 Sarcasm and satire can fool even advanced models.",
#     "tech stack": "💻 Python, scikit-learn, Streamlit, joblib, and Claude via OpenRouter API.",
#     "goal": "🎯 To demonstrate how machine learning can help fight misinformation.",
#     "fake news": "🚨 Fake news is false info spread deliberately to mislead.",
#     "machine learning": "🧠 ML lets systems learn from data to make predictions.",
#     "languages": "🌍 Currently, only English is supported.",
#     "ai": "🤖 I'm an AI assistant trained using NLP techniques.",
#     "thank you": "🙏 You're welcome! Stay informed.",
#     "thanks": "😊 No problem! Happy to help."
# }


# try 4(with spinner struck at the top but working as expected connecting to claude)


# import streamlit as st
# import re
# import joblib
# import numpy as np
# from bs4 import BeautifulSoup
# import os
# import time
# import requests
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()
# API_KEY = os.getenv("OPENROUTER_API_KEY")

# # Load ML model and vectorizer
# model = joblib.load("fake_news_model.pkl")
# vectorizer = joblib.load("tfidf_vectorizer.pkl")
# threshold = joblib.load("optimal_threshold.pkl")

# # Claude API Function
# def fetch_claude_fact_check(user_text):
#     url = "https://openrouter.ai/api/v1/chat/completions"
#     headers = {
#         "Authorization": f"Bearer {API_KEY}",
#         "Content-Type": "application/json"
#     }
#     data = {
#         "model": "anthropic/claude-3-haiku",
#         "messages": [
#             {"role": "system", "content": "You are a helpful fact-checking assistant. When a news statement is likely fake, provide what might be true instead."},
#             {"role": "user", "content": f"Fact-check this news: {user_text}. What might be true instead?"}
#         ]
#     }
#     try:
#         response = requests.post(url, headers=headers, json=data, timeout=10)
#         if response.status_code == 200:
#             return response.json()['choices'][0]['message']['content']
#         else:
#             return "⚠️ Claude API failed to respond."
#     except Exception as e:
#         return f"⚠️ Claude API error: {str(e)}"

# # Response mapping
# def get_custom_response(text):
#     text = text.lower().strip()
#     for keyword, response in custom_responses.items():
#         if keyword in text:
#             return response
#     greetings = ["hi", "hello", "hey", "yo", "hola"]
#     if any(greet in text for greet in greetings):
#         return "👋 Hello! I'm here to help you detect fake news or answer questions about the model."
#     return None

# # Preprocess input
# def extract_title_content(text):
#     lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
#     meaningful = [l for l in lines if not re.search(r'\b(fake|real|check|news|bot|tell me|is this)\b', l, re.IGNORECASE)]
#     title = meaningful[0] if meaningful else ""
#     content = " ".join(meaningful[1:]) if len(meaningful) > 1 else ""
#     return title, content

# def clean_html_to_text(html):
#     try:
#         return BeautifulSoup(html, "html.parser").get_text(separator="\n").strip()
#     except:
#         return html

# # Session and UI setup
# st.set_page_config(page_title="Fake News Detector", page_icon="🧠")
# st.title("📰 Fake News Detector Chatbot")
# st.markdown("""
#     <div style='background-color:red; padding: 10px; border-left: 5px solid #ffa500; border-radius: 5px;'>
#         ⚠️ <strong>Disclaimer:</strong> This tool is for educational use only. Always verify important news independently.
#     </div>
# """, unsafe_allow_html=True)

# # Sidebar downloads
# with st.sidebar:
#     st.header("📁 Downloads")
#     def generate_chat_log():
#         return "\n\n".join([f"{'You' if msg['role']=='user' else 'Bot'}: {msg['content']}" for msg in st.session_state.get("messages", [])])
#     st.download_button("📥 Download Chat Log", generate_chat_log(), "chat_history.txt")

# # State messages
# if "messages" not in st.session_state:
#     st.session_state.messages = []
# if not st.session_state.messages:
#     st.chat_message("assistant").markdown("👋 Hi! I'm here to help detect fake news. Paste a news article or ask about the model!")

# # User input
# user_input = st.chat_input("Paste your news article or ask a question...")
# if user_input:
#     st.session_state.messages.append({"role": "user", "content": user_input})
#     title, content = extract_title_content(user_input)
#     combined = f"{title} {content}".strip()

#     if len(combined.split()) < 6:
#         response = get_custom_response(user_input) or "❓ I'm not sure how to respond to that. Try rephrasing or provide a fuller article."
#         st.session_state.messages.append({"role": "assistant", "content": response})
#     else:
#         vec = vectorizer.transform([combined])
#         prob = model.predict_proba(vec)[0][1]
#         prediction = "FAKE" if prob >= threshold else "REAL"
#         confidence = round(prob if prediction == "FAKE" else 1 - prob, 2)

#         if prediction == "REAL":
#             html_response = f"""
#                 <div style='background-color:#d4edda; padding: 10px; border-left: 5px solid #28a745; border-radius: 5px;'>
#                     📢 <strong style='color:#155724;'>This news appears to be real</strong><br><br>
#                     <span style='color:#1b2e1f;'>🔍 <strong>Confidence:</strong> {round(confidence * 100, 2)}%</span>
#                 </div>
#             """
#         else:
#             spinner_placeholder = st.empty()
#             with spinner_placeholder.container():
#                 with st.spinner("🧠 Thinking about what might be true instead..."):
#                     time.sleep(2)
#                     fact_correction = fetch_claude_fact_check(user_input)
#             spinner_placeholder.empty()

#             html_response = f"""
#                 <div style='background-color:#f8d7da; padding: 10px; border-left: 5px solid #dc3545; border-radius: 5px;'>
#                     🚨 <strong style='color:#721c24;'>This news appears to be fake</strong><br><br>
#                     <span style='color:#4f1216;'>🔍 <strong>Confidence:</strong> {round(confidence * 100, 2)}%</span><br><br>
#                     <div style='background-color:#1e1e1e; color:#f1f1f1; padding:10px; border-radius:10px; font-family:monospace;'>
#                         <strong>What might be true:</strong><br>
#                         {fact_correction}
#                     </div>
#                 </div>
#             """

#         st.session_state.messages.append({"role": "assistant", "content": html_response})

# # Display messages
# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         if msg["role"] == "assistant" and ("appears to be" in msg["content"] or "<div" in msg["content"]):
#             st.markdown(msg["content"], unsafe_allow_html=True)
#         else:
#             st.markdown(msg["content"])

# # FAQ responses
# theme_emoji = "🧠"
# custom_responses = {
#     "how": f"{theme_emoji} I analyze the text using TF-IDF and classify it as real or fake using a Logistic Regression model.",
#     "model": "🔍 I'm powered by Logistic Regression — it's interpretable and effective for text classification.",
#     "data": "📚 The model was trained on a dataset of real and fake news articles from known sources.",
#     "tfidf": "📊 TF-IDF stands for Term Frequency-Inverse Document Frequency. It helps highlight important words.",
#     "logistic regression": "📈 Logistic Regression is a statistical model used for binary classification.",
#     "accuracy": "✅ The model achieves ~92% accuracy on test data.",
#     "trust": "🤔 Always double-check news from multiple trusted sources.",
#     "confidence": "📏 Confidence indicates how sure the model is about its prediction.",
#     "sarcasm": "😅 Sarcasm and satire can fool even advanced models.",
#     "tech stack": "💻 Python, scikit-learn, Streamlit, joblib, and Claude via OpenRouter API.",
#     "goal": "🎯 To demonstrate how machine learning can help fight misinformation.",
#     "fake news": "🚨 Fake news is false info spread deliberately to mislead.",
#     "machine learning": "🧠 ML lets systems learn from data to make predictions.",
#     "languages": "🌍 Currently, only English is supported.",
#     "ai": "🤖 I'm an AI assistant trained using NLP techniques.",
#     "thank you": "🙏 You're welcome! Stay informed.",
#     "thanks": "😊 No problem! Happy to help."
# }

# try 5 (fixing spinner)
# import streamlit as st
# import re
# import joblib
# import numpy as np
# import requests
# from bs4 import BeautifulSoup
# import os
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()
# API_KEY = os.getenv("OPENROUTER_API_KEY")


# def detect_intent(text):
#     candidate_labels = ["question", "news", "greeting", "thank you", "sarcasm"]
#     result = classifier(text, candidate_labels, multi_label=False)
#     return result["labels"][0]  # Return the most likely intent


# # Load ML model and vectorizer
# model = joblib.load("fake_news_model.pkl")
# vectorizer = joblib.load("tfidf_vectorizer.pkl")
# threshold = joblib.load("optimal_threshold.pkl")

# # Claude API
# def fetch_claude_fact_check(user_text):
#     url = "https://openrouter.ai/api/v1/chat/completions"
#     headers = {
#         "Authorization": f"Bearer {API_KEY}",
#         "Content-Type": "application/json"
#     }
#     data = {
#         "model": "anthropic/claude-3-haiku",
#         "messages": [
#             {"role": "system", "content": "You are a helpful fact-checking assistant. When a news statement is likely fake, provide what might be true instead."},
#             {"role": "user", "content": f"Fact-check this news: {user_text}. What might be true instead?"}
#         ]
#     }
#     try:
#         response = requests.post(url, headers=headers, json=data, timeout=10)
#         if response.status_code == 200:
#             return response.json()['choices'][0]['message']['content']
#         else:
#             return "⚠️ Claude API failed to respond."
#     except Exception as e:
#         return f"⚠️ Claude API error: {str(e)}"

# # Predefined response helper
# theme_emoji = "🧠"
# custom_responses = {
#     # Greetings
#     "hi": "👋 Hello! I'm your fake news detection assistant. Paste any article or headline, and I'll help you verify it.",
#     "hello": "👋 Hi there! I'm here to detect fake news or answer your questions.",
#     "hey": "👋 Hey! Ready to help you check any news.",
#     "greetings": "🙌 Greetings! What would you like to check today?",
#     "good morning": "🌅 Good morning! Let’s check some news together.",
#     "good afternoon": "☀️ Good afternoon! What news would you like help with?",
#     "good evening": "🌆 Good evening! I'm ready to detect fake news for you.",

#     # Farewells
#     "bye": "👋 Goodbye! Stay safe and informed.",
#     "goodbye": "👋 Take care! Always verify the news.",
#     "see you": "👀 See you later!",
#     "exit": "🔚 Exiting the session. Come back anytime!",

#     # Tool explanations
#     "how": "🧠 I analyze the text using TF-IDF and classify it as real or fake using a Logistic Regression model.",
#     "how does it work": "🧠 I use text features and a trained model to detect patterns typical of fake or real news.",
#     "model": "🔍 I'm powered by Logistic Regression — it's interpretable and effective for text classification.",
#     "data": "📚 The model was trained on a dataset of real and fake news articles from trusted and untrusted sources.",
#     "tfidf": "📊 TF-IDF stands for Term Frequency-Inverse Document Frequency. It highlights significant words in the text.",
#     "logistic regression": "📈 Logistic Regression is a statistical model used for binary classification tasks like this.",
#     "accuracy": "✅ The model achieves ~99% accuracy on test data, though real-world performance can vary.",
#     "confidence": "📏 Confidence reflects how sure the model is about the prediction — closer to 100% means more certain.",
#     "threshold": "🎚️ I use an optimized threshold to decide whether something is real or fake based on prediction probabilities.",

#     # Purpose and usage
#     "goal": "🎯 To demonstrate how machine learning can help fight misinformation in real time.",
#     "why": "❓ To help people verify the news they encounter online and avoid falling for misinformation.",
#     "who are you": "🤖 I'm an AI assistant built to detect fake news and explain how I work.",
#     "what can you do": "🛠️ I can help classify news articles and answer questions about how I work.",
#     "tech stack": "💻 Python, scikit-learn, Streamlit, joblib, and Claude API for reasoning.",
#     "languages": "🌍 Currently, I only support English news articles or questions.",
#     "fake news": "🚨 Fake news is false information spread with the intent to mislead. My job is to flag it.",
#     "real news": "📢 Real news comes from verified sources and aligns with factual reporting standards.",
#     "machine learning": "🧠 Machine learning lets systems learn patterns from data to make intelligent predictions.",
#     "ai": "🤖 I'm a type of AI assistant trained using Natural Language Processing and machine learning.",
#     "satire": "🗞️ Satire may look fake but is often intended for humor — it's tricky even for ML models.",
#     "sarcasm": "😅 Sarcasm and satire can sometimes fool detection models.",
#     "news sources": "📰 I don't check news sources directly but analyze text patterns. For sources, use tools like GNews or Snopes.",
    
#     # Clarifications and misunderstandings
#     "you are wrong": "😕 I may not be perfect. Please double-check with official news or fact-checking sites.",
#     "i disagree": "🧠 It's healthy to be skeptical. Always verify important news using trusted sources.",
#     "are you sure": "📏 I give predictions based on what I learned — use this as a guide, not absolute truth.",
#     "can i trust you": "🤔 Always double-check. I'm a helpful tool but not a substitute for human judgment.",

#     # User gratitude / kindness
#     "thank you": "🙏 You're welcome! Stay informed and skeptical.",
#     "thanks": "😊 No problem! I'm here to help.",
#     "you are helpful": "🙌 I'm glad I could help!",
#     "awesome": "🎉 Yay! Let me know what else you'd like to check.",

#     # Fun responses
#     "joke": "😄 Why did the fake news article go to therapy? It had identity issues.",
#     "bored": "🎲 Try giving me a news headline — I’ll check if it’s real or fake!",
#     "help": "🆘 Paste a news headline or paragraph, and I’ll help you analyze it. You can also ask how I work!"
# }

# def get_custom_response(user_input):
#     user_input = user_input.lower()
#     best_match = None
#     highest_score = 0

#     for key, response in custom_responses.items():
#         # Use regex to match whole words (case-insensitive)
#         pattern = r'\b' + re.escape(key) + r'\b'
#         if re.search(pattern, user_input):
#             score = len(key)
#             if score > highest_score:
#                 highest_score = score
#                 best_match = response

#     return best_match

# def extract_title_content(text):
#     lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
#     meaningful = [l for l in lines if not re.search(r'\b(fake|real|check|news|bot|tell me|is this)\b', l, re.IGNORECASE)]
#     title = meaningful[0] if meaningful else ""
#     content = " ".join(meaningful[1:]) if len(meaningful) > 1 else ""
#     return title, content

# # Page config
# st.set_page_config(page_title="Fake News Detector", page_icon="🧠")
# st.title("📰 Fake News Detector Chatbot")
# st.markdown("""
#     <div style='background-color:red; padding: 10px; border-left: 5px solid #ffa500; border-radius: 5px; color:white;'>
#         ⚠️ <strong>Disclaimer:</strong> This tool is for educational use only. Always verify important news independently.
#     </div>
# """, unsafe_allow_html=True)

# def is_question_or_keyword(text):
#     text = text.lower().strip()
#     question_keywords = [
#         "how", "what", "why", "where", "when", "can you", "who", "do you", "is this", "are you", "could you"
#     ]
#     # Detect question by structure or punctuation
#     if "?" in text or any(text.startswith(q) for q in question_keywords):
#         return True
    
#     # Check against predefined custom keywords
#     for key in custom_responses.keys():
#         if key in text:
#             return True
    
#     return False


# # Sidebar
# with st.sidebar:
#     st.header("📁 Downloads")
#     def generate_chat_log():
#         return "\n\n".join([f"{'You' if msg['role']=='user' else 'Bot'}: {msg['content']}" for msg in st.session_state.get("messages", [])])
#     st.download_button("📥 Download Chat Log", generate_chat_log(), "chat_history.txt")
#     st.markdown("""---""")

#     st.markdown("""
#     💡 **Tip:** To chat with the model or greetings and things like accuracy, algorithm, etc., prefix your question with: **`faq:`**  
#     Example:  
#     `faq: what is the accuracy?`
#     """)

#     st.markdown("""---""")

# # Session state init
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display existing messages
# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         if msg["role"] == "assistant" and ("<div" in msg["content"]):
#             st.markdown(msg["content"], unsafe_allow_html=True)
#         else:
#             st.markdown(msg["content"])

# # Chat input
# user_input = st.chat_input("Paste your news article or ask a question...")

# if user_input:
#     # Add user input first
#     st.session_state.messages.append({"role": "user", "content": user_input})
#     with st.chat_message("user"):
#         st.markdown(user_input)

#     # Check for FAQ mode
#     if user_input.lower().startswith("faq:"):
#         stripped_question = user_input[4:].strip()
#         response = get_custom_response(stripped_question) or "❓ Sorry, I don't have a specific answer for that FAQ."
#         with st.chat_message("assistant"):
#             st.markdown(response)
#         st.session_state.messages.append({"role": "assistant", "content": response})
    
#     else:
#         # Process as news
#         title, content = extract_title_content(user_input)
#         combined = f"{title} {content}".strip()
#         vec = vectorizer.transform([combined])
#         prob = model.predict_proba(vec)[0][1]
#         prediction = "FAKE" if prob >= threshold else "REAL"
#         confidence = round(prob if prediction == "FAKE" else 1 - prob, 2)

#         with st.chat_message("assistant"):
#                 with st.spinner("🧠 Thinking about what might be true instead..."):
#                     fact = fetch_claude_fact_check(user_input)

#                 if prediction == "REAL":
#                     html_response = f"""
#                         <div style='background-color:#d4edda; padding: 10px; border-left: 5px solid #28a745; border-radius: 5px;'>
#                             📢 <strong style='color:#155724;'>This news appears to be real</strong><br><br>
#                             <span style='color:#1b2e1f;'>🔍 <strong>Confidence:</strong> {round(confidence * 100, 2)}%</span>
#                         </div>
#                     """
#                 else:
#                     html_response = f"""
#                         <div style='background-color:#f8d7da; padding: 10px; border-left: 5px solid #dc3545; border-radius: 5px;'>
#                             🚨 <strong style='color:#721c24;'>This news appears to be fake</strong><br><br>
#                             <span style='color:#4f1216;'>🔍 <strong>Confidence:</strong> {round(confidence * 100, 2)}%</span><br><br>
#                             <div style='background-color:#1e1e1e; color:#f1f1f1; padding:10px; border-radius:10px; font-family:monospace;'>
#                                 <strong>What might be true:</strong><br>
#                                 {fact}
#                             </div>
#                         </div>
#                     """
#                 st.markdown(html_response, unsafe_allow_html=True)
#                 st.session_state.messages.append({"role": "assistant", "content": html_response})
import streamlit as st
import joblib
import os
import re
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load Claude/OpenRouter API key
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

# Load ML components
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
threshold = joblib.load("optimal_threshold.pkl")

# --- Claude Fact-checking ---
def fetch_claude_fact_check(user_text):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "anthropic/claude-3-haiku",
        "messages": [
            {"role": "system", "content": "You are a helpful fact-checking assistant. When a news statement is likely fake, provide what might be true instead."},
            {"role": "user", "content": f"Fact-check this news: {user_text}. What might be true instead?"}
        ]
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return "⚠️ Claude API failed to respond."
    except Exception as e:
        return f"⚠️ Claude API error: {str(e)}"

# --- Predefined FAQ ---
custom_responses = {
    "hi": "👋 Hello! I'm your fake news detection assistant. Paste any article or headline, and I'll help you verify it.",
    "hello": "👋 Hi there! I'm here to detect fake news or answer your questions.",
    "bye": "👋 Goodbye! Stay safe and informed.",
    "model": "🔍 I'm powered by Logistic Regression — it's interpretable and effective for text classification.",
    "accuracy": "✅ The model achieves ~99% accuracy on test data, though real-world performance can vary.",
    "help": "🆘 Paste a news headline or paragraph, and I’ll help you analyze it. You can also ask how I work!"
}

def get_custom_response(user_input):
    user_input = user_input.lower()
    for key, response in custom_responses.items():
        if re.search(rf'\b{re.escape(key)}\b', user_input):
            return response
    return None

# --- Extract title and content from user input ---
def extract_title_content(text):
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    meaningful = [l for l in lines if not re.search(r'\b(fake|real|check|news|bot|tell me|is this)\b', l, re.IGNORECASE)]
    title = meaningful[0] if meaningful else ""
    content = " ".join(meaningful[1:]) if len(meaningful) > 1 else ""
    return title, content

# --- Page Setup ---
st.set_page_config(page_title="Fake News Detector", page_icon="🧠")
st.title("📰 Fake News Detector Chatbot")

st.markdown("""
    <div style='background-color:red; padding: 10px; border-left: 5px solid #ffa500; border-radius: 5px; color:white;'>
        ⚠️ <strong>Disclaimer:</strong> This tool is for educational use only. Always verify important news independently.
    </div>
""", unsafe_allow_html=True)

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.header("📁 Downloads")

    def generate_chat_log():
        log = []
        for msg in st.session_state.get("messages", []):
            role = "You" if msg["role"] == "user" else "Bot"
            text = BeautifulSoup(msg["content"], "html.parser").get_text().strip()
            log.append(f"{role}:\n{text}\n" + "-" * 50)
        return "\n\n".join(log)

    st.download_button(
        label="📥 Download Chat Log",
        data=generate_chat_log(),
        file_name="chat_history.txt",
        mime="text/plain"
    )

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("""
    💡 **Tip:** To ask questions about how I work, prefix with `faq:`  
    Example: `faq: what is the accuracy?`
    """)

# --- Display Previous Messages ---
for i, msg in enumerate(st.session_state.messages):
    if "timestamp" not in msg or not msg["timestamp"]:
        st.session_state.messages[i]["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html="<div" in msg["content"])
        try:
            t = datetime.strptime(msg["timestamp"], "%Y-%m-%d %H:%M:%S").strftime("%b %d %Y, %H:%M")
        except:
            t = msg["timestamp"]
        st.caption(f"🕒 {t}")

# --- Input Box ---
user_input = st.chat_input("Paste your news article or ask a question...")

if user_input:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Log user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": timestamp
    })

    with st.chat_message("user"):
        st.markdown(user_input)
        t = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").strftime("%b %d %Y, %H:%M")
        st.caption(f"🕒 {t}")

    # Handle FAQ
    if user_input.lower().startswith("faq:"):
        q = user_input[4:].strip()
        answer = get_custom_response(q) or "❓ Sorry, I don't understand. Try pasting a news headline or use 'faq:' for help."

        assistant_msg = {
            "role": "assistant",
            "content": answer,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        st.session_state.messages.append(assistant_msg)

        with st.chat_message("assistant"):
            st.markdown(answer)
            t = datetime.strptime(assistant_msg['timestamp'], "%Y-%m-%d %H:%M:%S").strftime("%b %d %Y, %H:%M")
            st.caption(f"🕒 {t}")

    else:
        # Handle News Check
        title, content = extract_title_content(user_input)
        combined = f"{title} {content}".strip()
        vec = vectorizer.transform([combined])
        prob = model.predict_proba(vec)[0][1]
        prediction = "FAKE" if prob >= threshold else "REAL"
        confidence = round(prob if prediction == "FAKE" else 1 - prob, 2)

        with st.spinner("🧠 Thinking about the news..."):
            fact = fetch_claude_fact_check(user_input)

        if prediction == "REAL":
            html_response = f"""
                <div style='background-color:#d4edda; padding: 10px; border-left: 5px solid #28a745; border-radius: 5px;'>
                    📢 <strong style='color:#155724;'>This news appears to be real</strong><br><br>
                    <span style='color:#1b2e1f;'>🔍 <strong>Confidence:</strong> {round(confidence * 100, 2)}%</span>
                </div>
            """
        else:
            html_response = f"""
                <div style='background-color:#f8d7da; padding: 10px; border-left: 5px solid #dc3545; border-radius: 5px;'>
                    🚨 <strong style='color:#721c24;'>This news appears to be fake</strong><br><br>
                    <span style='color:#4f1216;'>🔍 <strong>Confidence:</strong> {round(confidence * 100, 2)}%</span><br><br>
                    <div style='background-color:#1e1e1e; color:#f1f1f1; padding:10px; border-radius:10px; font-family:monospace;'>
                        <strong>What might be true:</strong><br>
                        {fact}
                    </div>
                </div>
            """

        assistant_msg = {
            "role": "assistant",
            "content": html_response,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        st.session_state.messages.append(assistant_msg)

        with st.chat_message("assistant"):
            st.markdown(html_response, unsafe_allow_html=True)
            t = datetime.strptime(assistant_msg['timestamp'], "%Y-%m-%d %H:%M:%S").strftime("%b %d %Y, %H:%M")
            st.caption(f"🕒 {t}")
