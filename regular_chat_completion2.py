import requests
import json
from time import sleep

import asyncio
import aiohttp
import json
import glob
import re
import pandas as pd


class AsyncChatModel:
    def __init__(self):
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.OPENROUTER_API_KEY = input("Please enter your Openrouter API key:\n")
        assert self.OPENROUTER_API_KEY.startswith("sk"), "API key must start with 'sk'"

    async def chat(self, session, user_input, model, semaphore, **kwargs):
        messages = [{'role': 'user', 'content': user_input}]

        headers = {
            "Authorization": f"Bearer {self.OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://google.com",
            "X-Title": "My Colab Project",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            **kwargs,
        }

        MAX_RETRIES = 5
        RETRY_DELAY = 5.0 

        async with semaphore:
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    async with session.post(self.url, json=payload, headers=headers) as response:
                        if response.status == 429 or response.status >= 500:
                            print(f"[attempt {attempt}/{MAX_RETRIES}] HTTP {response.status}")
                        else:
                            response.raise_for_status()
                            data = await response.json()
                            break
                except asyncio.TimeoutError:
                    print(f"[attempt {attempt}/{MAX_RETRIES}] Network timeout")
                except aiohttp.ClientResponseError as e:
                    print(f"Non-retryable HTTP error: {e.status} {e.message}")
                    return {}
                except Exception as e:
                    print(f"[attempt {attempt}/{MAX_RETRIES}] Unexpected error: {e}")

                if attempt == MAX_RETRIES:
                    print("Max retries exceeded")
                    return {}
                
                await asyncio.sleep(RETRY_DELAY) 
            
            try:
                return data['choices'][0]['message']['content']
            except (KeyError, IndexError, TypeError, UnboundLocalError) as e:
                print(f"Unexpected response format: {e}")
                return {}


class ChatModel:
    def __init__(self, deployment: str):
        self.url = "https://openrouter.ai/api/v1/chat/completions"

        self.model = deployment
        self.name = deployment

        self.OPENROUTER_API_KEY = input("Please enter your Openrouter API key:\n")
        assert self.OPENROUTER_API_KEY.startswith("sk")

        response = requests.get(
            url="https://openrouter.ai/api/v1/key",
            headers={"Authorization": f"Bearer {self.OPENROUTER_API_KEY}"},
        )
        print(json.dumps(response.json(), indent=2))

    def chat(self, user_input, **kwargs):
        messages = [dict(role='user', content=user_input)]

        headers = {
            "Authorization": f"Bearer {self.OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://google.com",
            "X-Title": "My Colab Project",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            **kwargs,
        }

        MAX_RETRIES = 5
        RETRY_DELAY = 5.0 

        try:
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    response = requests.post(self.url, json=payload, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                    break  
                except (requests.Timeout) as e:
                    print(f"[attempt {attempt}/{MAX_RETRIES}] Network error: {e}")
                except requests.HTTPError as e:
                    if e.response.status_code == 429 or e.response.status_code >= 500:
                        print(f"[attempt {attempt}/{MAX_RETRIES}] HTTP {e.response.status_code}: {e}")
                    else:
                        print(f"Non-retryable HTTP error: {e.response.status_code} {e}")
                        return {}
                except Exception as e:
                    print(f"[attempt {attempt}/{MAX_RETRIES}] Unexpected error: {e}")

                if attempt == MAX_RETRIES:
                    print("Max retries exceeded")
                    return {}
                sleep(RETRY_DELAY)
        except Exception as e:
            print(f"Error calling OpenRouter: {e}")
            return {}

        try:
            data = data['choices'][0]['message']['content']
        except (KeyError, IndexError, TypeError) as e:
            print(f"Unexpected response format: {e}")
            return {}
        return data
