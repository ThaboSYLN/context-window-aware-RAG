from google import genai
import os

# Initialize the client with your Gemini key
client = genai.Client(api_key="AIzaSyDn_5SHc2sH8eke_c-6LRSI1BixL9GZ3kc")

try:
    print("--- Sending Gemini Request ---")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Say 'Gemini is working!'"
    )
    
    print("AI Response:", response.text)
    print("--- Test Successful ---")

except Exception as e:
    print(f"--- Error Encountered ---")
    print(f"Message: {e}")

