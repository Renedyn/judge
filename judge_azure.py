import math
import httpx
import asyncio
import requests
import json

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

class OpenAiJudge:
    def __init__(self, deployment: str, prompt_templates: dict):
        self.model = deployment
        self.prompt_templates = prompt_templates
        self.name = deployment

        self.OPENROUTER_API_KEY = input("Please enter your Openrouter API key:\n")
        assert self.OPENROUTER_API_KEY.startswith("sk")

        response = requests.get(
            url="https://openrouter.ai/api/v1/key",
            headers={"Authorization": f"Bearer {self.OPENROUTER_API_KEY}"},
        )
        print(json.dumps(response.json(), indent=2))

    async def judge(self, client, metric, **kwargs):
        messages = [dict(role='user', content=self.prompt_templates[metric].format(**kwargs))]
        logprobs = await self.logprob_probs(client, messages)
        return self._aggregate_0_100_score(logprobs)

    def judge_batch(self, metric, batch: list[dict]) -> list[float | None]:
        async def _run():
            async with httpx.AsyncClient(timeout=60.0) as client:
                return await asyncio.gather(
                    *(self.judge(client, metric, **kwargs) for kwargs in batch)
                )
        return asyncio.run(_run())

    async def logprob_probs(self, client, messages) -> dict:
        headers = {
            "Authorization": f"Bearer {self.OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://google.com",
            "X-Title": "My Colab Project",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "provider": {"only": ["alibaba"]},
            "logprobs": True,
            "max_tokens": 1,
            "top_logprobs": 5,
            "temperature": 1,
            "seed": 0,
            "reasoning": {"enabled": False},
        }

        MAX_RETRIES = 5
        RETRY_DELAY = 5.0 

        try:
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    response = await client.post(OPENROUTER_URL, json=payload, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                    break  
                except (httpx.TimeoutException, httpx.TransportError) as e:
                    print(f"[attempt {attempt}/{MAX_RETRIES}] Network error: {e}")
                except httpx.HTTPStatusError as e:
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
                await asyncio.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"Error calling OpenRouter: {e}")
            return {}

        try:
            logprobs_data = data['choices'][0]['logprobs']['content'][0]['top_logprobs']
        except (KeyError, IndexError, TypeError) as e:
            print(f"Unexpected response format: {e}")
            return {}

        result = {}
        for item in logprobs_data:
            token = item.get('token', '')
            logprob = item.get('logprob', 0.0)
            try:
                result[token] = float(math.exp(logprob))
            except Exception:
                continue
        return result

    def _aggregate_0_100_score(self, score: dict) -> float | None:
        total = 0.0
        sum_ = 0.0
        for key, val in score.items():
            try:
                int_key = int(key)
            except ValueError:
                continue
            if 0 <= int_key <= 5:
                sum_ += int_key * val
                total += val
        if total < 0.25:
            return None
        return 20 * sum_ / total