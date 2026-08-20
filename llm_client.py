import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def get_llm_client_and_model(provider=None):
    """
    Returns (client, model_name, provider_name) for requested provider.
    Supported providers: 'groq', 'gemini', 'openrouter'.
    """
    prov = (provider or os.getenv("LLM_PROVIDER", "groq")).lower()

    # 1. GROQ CLOUD (Primary Ultra-Fast Model Engine)
    if prov == "groq":
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=groq_key
            )
            return client, "openai/gpt-oss-120b", "groq"

    # 2. GOOGLE GEMINI FLASH (Backup 1)
    if prov == "gemini":
        gem_key = os.getenv("GEMINI_API_KEY")
        if gem_key:
            client = OpenAI(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=gem_key
            )
            return client, "gemini-3.6-flash", "gemini"

    # 3. OPENROUTER (Backup 2)
    or_key = os.getenv("OPENROUTER_API_KEY")
    if or_key:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=or_key
        )
        return client, "anthropic/claude-3-haiku", "openrouter"

    # Fallback to Groq
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_key
        )
        return client, "openai/gpt-oss-120b", "groq"

    raise ValueError("No valid LLM API key found in .env! Please set GROQ_API_KEY or GEMINI_API_KEY.")

def generate_chat_completion(messages, temperature=0.1, max_tokens=1000):
    """
    Executes LLM completion with automatic multi-provider fallback:
    Groq -> Gemini -> OpenRouter.
    Completely isolated from console encoding bugs.
    """
    providers_to_try = []
    
    primary = os.getenv("LLM_PROVIDER", "groq").lower()
    providers_to_try.append(primary)
    
    for fallback in ["groq", "gemini", "openrouter"]:
        if fallback not in providers_to_try:
            providers_to_try.append(fallback)

    last_error = None
    for prov in providers_to_try:
        try:
            client, model_name, used_prov = get_llm_client_and_model(prov)
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            content = None
            if response and response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                
            if content:
                if "<think>" in content and "</think>" in content:
                    content = content.split("</think>")[-1].strip()
                # Return content immediately without any console print hazards
                return content.strip(), used_prov

        except Exception as e:
            # Store API error and try next provider
            last_error = e

    raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")
