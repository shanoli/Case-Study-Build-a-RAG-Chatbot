"""
Prompt templates for RAG and classification
"""

# System prompt for RAG - Multi-lingual aware
RAG_SYSTEM_PROMPT = """You are a helpful customer support assistant for TechGear, an electronics retailer.

IMPORTANT GUIDELINES:
1. **Multi-lingual Support**: You can understand and respond in ANY language the user speaks (English, Hindi, Tamil, Telugu, Spanish, French, etc.)
2. **Language Matching**: Always respond in the SAME language the user is using
3. **Context-based Answers**: Use only the provided context to answer questions
4. **Product Focus**: Specialize in our products: SmartWatch Pro X, Wireless Earbuds Elite, Power Bank Ultra
5. **Accuracy**: If information is not in the context, politely say you don't have that information
6. **Tone**: Be friendly, helpful, and professional
7. **Pricing**: Always mention prices in Indian Rupees (₹)

AVAILABLE INFORMATION:
- Product details (features, prices, warranty)
- Return policy
- Support contact information

Remember: Detect the user's language and respond accordingly. If they ask in Hindi, respond in Hindi. If they ask in English, respond in English.
"""

# User prompt template for RAG
RAG_USER_PROMPT = """Based on the following context, please answer the user's question.

CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER (in the same language as the question):"""

# Fallback response when no context is found
FALLBACK_RESPONSE = {
    "en": "I don't have specific information about that in our knowledge base. I can help you with questions about our products (SmartWatch Pro X, Wireless Earbuds Elite, Power Bank Ultra), pricing, warranty, return policy, or support contact. What would you like to know?",
    "hi": "मुझे अपने ज्ञान आधार में इसके बारे में विशिष्ट जानकारी नहीं है। मैं आपको हमारे उत्पादों (SmartWatch Pro X, Wireless Earbuds Elite, Power Bank Ultra), मूल्य निर्धारण, वारंटी, रिटर्न पॉलिसी या सपोर्ट संपर्क के बारे में प्रश्नों में मदद कर सकता हूं। आप क्या जानना चाहेंगे?",
    "ta": "எங்கள் அறிவுத் தளத்தில் இதைப் பற்றிய குறிப்பிட்ட தகவல் என்னிடம் இல்லை. எங்கள் தயாரிப்புகள் (SmartWatch Pro X, Wireless Earbuds Elite, Power Bank Ultra), விலை நிர்ணயம், உத்தரவாதம், திரும்பப் பெறும் கொள்கை அல்லது ஆதரவு தொடர்பு பற்றிய கேள்விகளில் நான் உங்களுக்கு உதவ முடியும். நீங்கள் என்ன தெரிந்து கொள்ள விரும்புகிறீர்கள்?",
    "te": "మా జ్ఞాన స్థావరంలో దీని గురించి నాకు నిర్దిష్ట సమాచారం లేదు. మా ఉత్పత్తులు (SmartWatch Pro X, Wireless Earbuds Elite, Power Bank Ultra), ధర నిర్ణయం, వారంటీ, రిటర్న్ పాలసీ లేదా సపోర్ట్ కాంటాక్ట్ గురించిన ప్రశ్నలలో నేను మీకు సహాయం చేయగలను. మీరు ఏమి తెలుసుకోవాలనుకుంటున్నారు?",
    "es": "No tengo información específica sobre eso en nuestra base de conocimientos. Puedo ayudarte con preguntas sobre nuestros productos (SmartWatch Pro X, Wireless Earbuds Elite, Power Bank Ultra), precios, garantía, política de devolución o contacto de soporte. ¿Qué te gustaría saber?",
    "fr": "Je n'ai pas d'informations spécifiques à ce sujet dans notre base de connaissances. Je peux vous aider avec des questions sur nos produits (SmartWatch Pro X, Wireless Earbuds Elite, Power Bank Ultra), les prix, la garantie, la politique de retour ou le contact d'assistance. Que souhaitez-vous savoir ?"
}

# Greeting messages - Multi-lingual
GREETING_RESPONSES = {
    "en": "Hello! Welcome to TechGear support. I can help you with information about our products (SmartWatch Pro X, Wireless Earbuds Elite, Power Bank Ultra), pricing, warranty, returns, and support. How can I assist you today?",
    "hi": "नमस्ते! TechGear सपोर्ट में आपका स्वागत है। मैं आपको हमारे उत्पादों (SmartWatch Pro X, Wireless Earbuds Elite, Power Bank Ultra), मूल्य निर्धारण, वारंटी, रिटर्न और सपोर्ट के बारे में जानकारी देने में मदद कर सकता हूं। मैं आज आपकी कैसे मदद कर सकता हूं?",
    "ta": "வணக்கம்! TechGear ஆதரவுக்கு வரவேற்கிறோம். எங்கள் தயாரிப்புகள் (SmartWatch Pro X, Wireless Earbuds Elite, Power Bank Ultra), விலை, உத்தரவாதம், திரும்பப் பெறுதல் மற்றும் ஆதரவு பற்றிய தகவல்களுடன் நான் உங்களுக்கு உதவ முடியும். இன்று நான் உங்களுக்கு எப்படி உதவ முடியும்?",
    "te": "హలో! TechGear సపోర్ట్‌కు స్వాగతం. మా ఉత్పత్తులు (SmartWatch Pro X, Wireless Earbuds Elite, Power Bank Ultra), ధర, వారంటీ, రిటర్న్స్ మరియు సపోర్ట్ గురించిన సమాచారంతో నేను మీకు సహాయం చేయగలను. ఈ రోజు నేను మీకు ఎలా సహాయం చేయగలను?",
    "es": "¡Hola! Bienvenido al soporte de TechGear. Puedo ayudarte con información sobre nuestros productos (SmartWatch Pro X, Wireless Earbuds Elite, Power Bank Ultra), precios, garantía, devoluciones y soporte. ¿Cómo puedo ayudarte hoy?",
    "fr": "Bonjour ! Bienvenue sur le support TechGear. Je peux vous aider avec des informations sur nos produits (SmartWatch Pro X, Wireless Earbuds Elite, Power Bank Ultra), les prix, la garantie, les retours et le support. Comment puis-je vous aider aujourd'hui ?"
}

# Escalation messages - Multi-lingual
ESCALATION_RESPONSES = {
    "en": "I'm not sure I understand that question. Could you please rephrase it? I can help you with:\n• Product information (SmartWatch, Earbuds, Power Bank)\n• Pricing and warranty details\n• Return policy\n• Support contact",
    "hi": "मुझे यकीन नहीं है कि मैं उस सवाल को समझता हूं। क्या आप कृपया इसे दोबारा बता सकते हैं? मैं आपकी मदद कर सकता हूं:\n• उत्पाद जानकारी (SmartWatch, Earbuds, Power Bank)\n• मूल्य और वारंटी विवरण\n• रिटर्न पॉलिसी\n• सपोर्ट संपर्क",
    "ta": "அந்தக் கேள்வியை நான் புரிந்துகொண்டதாக எனக்குத் தெரியவில்லை. தயவுசெய்து அதை மறுபரிசீலனை செய்ய முடியுமா? நான் உங்களுக்கு உதவ முடியும்:\n• தயாரிப்பு தகவல் (SmartWatch, Earbuds, Power Bank)\n• விலை மற்றும் உத்தரவாத விவரங்கள்\n• திரும்பப் பெறும் கொள்கை\n• ஆதரவு தொடர்பு",
    "te": "ఆ ప్రశ్న నాకు అర్థం కాలేదు. దయచేసి దాన్ని తిరిగి చెప్పగలరా? నేను మీకు సహాయం చేయగలను:\n• ఉత్పత్తి సమాచారం (SmartWatch, Earbuds, Power Bank)\n• ధర మరియు వారంటీ వివరాలు\n• రిటర్న్ పాలసీ\n• సపోర్ట్ సంప్రదింపు",
    "es": "No estoy seguro de entender esa pregunta. ¿Podrías reformularla? Puedo ayudarte con:\n• Información del producto (SmartWatch, Earbuds, Power Bank)\n• Detalles de precios y garantía\n• Política de devolución\n• Contacto de soporte",
    "fr": "Je ne suis pas sûr de comprendre cette question. Pourriez-vous la reformuler ? Je peux vous aider avec :\n• Informations sur les produits (SmartWatch, Earbuds, Power Bank)\n• Détails de prix et de garantie\n• Politique de retour\n• Contact d'assistance"
}

# Classifier Prompt (Unused but kept for import compatibility)
CLASSIFIER_PROMPT = """Classify the following user query into one of these categories:
1. products: Questions about products, features, specs, stock
2. returns: Questions about returns, refunds, exchanges
3. general: General questions, support, shipping, hours
4. greeting: Hellos, greetings, small talk

USER QUERY: {message}

Return only the category name."""

# Classification keywords for intent detection
INTENT_KEYWORDS = {
    "products": [
        "product", "products", "item", "items", "smartwatch", "earbuds", "power bank",
        "watch", "headphones", "charger", "buy", "purchase", "available", "stock",
        "उत्पाद", "घड़ी", "ईयरबड्स", "पावर बैंक",  # Hindi
        "தயாரிப்பு", "கடிகாரம்", "ஈயர்பட்ஸ்",  # Tamil
        "ఉత్పత్తి", "గడియారం", "ఇయర్‌బడ్స్",  # Telugu
        "producto", "reloj", "auriculares",  # Spanish
        "produit", "montre", "écouteurs"  # French
    ],
    "returns": [
        "return", "refund", "exchange", "replace", "money back", "cancel",
        "रिटर्न", "रिफंड", "वापसी",  # Hindi
        "திரும்ப", "பணத்தைத் திரும்பப் பெறுதல்",  # Tamil
        "తిరిగి", "రీఫండ్",  # Telugu
        "devolución", "reembolso",  # Spanish
        "retour", "remboursement"  # French
    ],
    "warranty": [
        "warranty", "guarantee", "coverage", "protection", "extended",
        "वारंटी", "गारंटी",  # Hindi
        "உத்தரவாதம்",  # Tamil
        "వారంటీ",  # Telugu
        "garantía",  # Spanish
        "garantie"  # French
    ],
    "support": [
        "support", "help", "contact", "customer service", "call", "email",
        "सपोर्ट", "मदद", "संपर्क",  # Hindi
        "ஆதரவு", "உதவி",  # Tamil
        "సపోర్ట్", "సహాయం",  # Telugu
        "soporte", "ayuda",  # Spanish
        "support", "aide"  # French
    ],
    "greeting": [
        "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
        "namaste", "नमस्ते", "हाय",  # Hindi
        "வணக்கம்", "ஹாய்",  # Tamil
        "నమస్తే", "హాయ్",  # Telugu
        "hola", "buenos días",  # Spanish
        "bonjour", "salut"  # French
    ]
}


def detect_language(text: str) -> str:
    """
    Detect language from text (simple heuristic)
    
    Args:
        text: Input text
    
    Returns:
        Language code (en, hi, ta, te, es, fr)
    """
    text_lower = text.lower()
    
    # Check for language-specific characters/keywords
    if any(char in text for char in ['नम', 'क्', 'हैं', 'मैं']):
        return "hi"
    elif any(char in text for char in ['வண', 'க்க', 'ம்']):
        return "ta"
    elif any(char in text for char in ['నమ', 'క్', 'లు']):
        return "te"
    elif any(word in text_lower for word in ['hola', 'buenos', 'gracias']):
        return "es"
    elif any(word in text_lower for word in ['bonjour', 'merci', 'salut']):
        return "fr"
    
    return "en"  # Default to English
