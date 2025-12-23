import requests
import time

def test_api():
    url = "http://127.0.0.1:8000/analyze"
    payload = {
        "jd_text": "We are looking for a software engineer with experience in Python, FastAPI, and Machine Learning. Must know Docker.",
        "resume_text": "I am a software engineer. I have 5 years of experience in Python and have built APIs using FastAPI. I also use Docker for deployment."
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("Status Code:", response.status_code)
        print("Response:", response.json())
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Give the server a moment to fully start if it hasn't yet
    time.sleep(5) 
    test_api()
