# custom_responses.py

custom_responses = {
    # Greetings
    "hii": "👋 Hello! I'm your fake news detection assistant. Paste any article or headline, and I'll help you verify it.",
    "hi": "👋 Hello! I'm your fake news detection assistant. Paste any article or headline, and I'll help you verify it.",
    "hello": "👋 Hi there! I'm here to detect fake news or answer your questions.",
    "hey": "👋 Hey! Ready to help you check any news.",
    "greetings": "🙌 Greetings! What would you like to check today?",
    "good morning": "🌅 Good morning! Let’s check some news together.",
    "good afternoon": "☀️ Good afternoon! What news would you like help with?",
    "good evening": "🌆 Good evening! I'm ready to detect fake news for you.",

    # Farewells
    "bye": "👋 Goodbye! Stay safe and informed.",
    "goodbye": "👋 Take care! Always verify the news.",
    "see you": "👀 See you later!",
    "exit": "🔚 Exiting the session. Come back anytime!",

    # Tool explanations
    "how": "🧠 I analyze the text using TF-IDF and classify it as real or fake using a Logistic Regression model.",
    "how does it work": "🧠 I use text features and a trained model to detect patterns typical of fake or real news.",
    "model": "🔍 I'm powered by Logistic Regression — it's interpretable and effective for text classification.",
    "data": "📚 The model was trained on a dataset of real and fake news articles from trusted and untrusted sources.",
    "tfidf": "📊 TF-IDF stands for Term Frequency-Inverse Document Frequency. It highlights significant words in the text.",
    "logistic regression": "📈 Logistic Regression is a statistical model used for binary classification tasks like this.",
    "accuracy": "✅ The model achieves ~99% accuracy on test data, though real-world performance can vary.",
    "confidence": "📏 Confidence reflects how sure the model is about the prediction — closer to 100% means more certain.",
    "threshold": "🎚️ I use an optimized threshold to decide whether something is real or fake based on prediction probabilities.",

    # Purpose and usage
    "goal": "🎯 To demonstrate how machine learning can help fight misinformation in real time.",
    "why": "❓ To help people verify the news they encounter online and avoid falling for misinformation.",
    "who are you": "🤖 I'm an AI assistant built to detect fake news and explain how I work.",
    "what can you do": "🛠️ I can help classify news articles and answer questions about how I work.",
    "tech stack": "💻 Python, scikit-learn, Streamlit, joblib, and Claude API for reasoning.",
    "languages": "🌍 Currently, I only support English news articles or questions.",
    "fake news": "🚨 Fake news is false information spread with the intent to mislead. My job is to flag it.",
    "real news": "📢 Real news comes from verified sources and aligns with factual reporting standards.",
    "machine learning": "🧠 Machine learning lets systems learn patterns from data to make intelligent predictions.",
    "ai": "🤖 I'm a type of AI assistant trained using Natural Language Processing and machine learning.",
    "satire": "🗞️ Satire may look fake but is often intended for humor — it's tricky even for ML models.",
    "sarcasm": "😅 Sarcasm and satire can sometimes fool detection models.",
    "news sources": "📰 I don't check news sources directly but analyze text patterns. For sources, use tools like GNews or Snopes.",
    
    # Clarifications and misunderstandings
    "you are wrong": "😕 I may not be perfect. Please double-check with official news or fact-checking sites.",
    "i disagree": "🧠 It's healthy to be skeptical. Always verify important news using trusted sources.",
    "are you sure": "📏 I give predictions based on what I learned — use this as a guide, not absolute truth.",
    "can i trust you": "🤔 Always double-check. I'm a helpful tool but not a substitute for human judgment.",

    # User gratitude / kindness
    "thank you": "🙏 You're welcome! Stay informed and skeptical.",
    "thankyou": "🙏 You're welcome! Stay informed and skeptical.",
    "thanks": "😊 No problem! I'm here to help.",
    "you are helpful": "🙌 I'm glad I could help!",
    "awesome": "🎉 Yay! Let me know what else you'd like to check.",

    # Fun responses
    "joke": "😄 Why did the fake news article go to therapy? It had identity issues.",
    "bored": "🎲 Try giving me a news headline — I’ll check if it’s real or fake!",
    "help": "🆘 Paste a news headline or paragraph, and I’ll help you analyze it. You can also ask how I work!"
}
