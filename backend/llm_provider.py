"""
Swappable LLM provider.

Set LLM_PROVIDER in .env to one of: "gemini", "groq", "cerebras"
Set the matching API key: GEMINI_API_KEY / GROQ_API_KEY / CEREBRAS_API_KEY

This keeps the RAG logic in main.py provider-agnostic — swapping providers
is a one-line .env change, not a code change.
"""

import os
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()


def call_llm(system_prompt: str, user_prompt: str) -> str:
    if PROVIDER == "gemini":
        return _call_gemini(system_prompt, user_prompt)
    elif PROVIDER == "groq":
        return _call_groq(system_prompt, user_prompt)
    elif PROVIDER == "cerebras":
        return _call_cerebras(system_prompt, user_prompt)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {PROVIDER}")


def _call_gemini(system_prompt: str, user_prompt: str) -> str:
    import google.generativeai as genai

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=system_prompt,
    )
    response = model.generate_content(user_prompt)
    return response.text


def _call_groq(system_prompt: str, user_prompt: str) -> str:
    from groq import Groq

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


def _call_cerebras(system_prompt: str, user_prompt: str) -> str:
    from cerebras.cloud.sdk import Cerebras

    client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.3-70b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content
