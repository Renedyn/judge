import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class TransformersJudge:
    def __init__(
        self,
        model: str,
        prompt_templates: str,
        name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.prompt_templates = prompt_templates
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=dtype,
            device_map=device,
        )
        self.model.eval()
        self.name = name

    def _build_prompt(self, metric, **kwargs) -> str:
        """Применяет chat-template модели к промпту."""
        text = self.prompt_templates[metric].format(**kwargs)
        messages = [{"role": "user", "content": text}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        return prompt

    @torch.inference_mode()
    def _get_logprobs(self, prompt: str) -> dict:
        """
        Прогоняет промпт через модель и возвращает распределение
        вероятностей для первого генерируемого токена.
        Возвращает dict: {token_string -> probability}
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model(**inputs)

        # logits для следующего токена после последнего входного
        # shape: (vocab_size,)
        next_token_logits = outputs.logits[0, -1, :]

        # Переводим в вероятности
        probs = torch.softmax(next_token_logits, dim=-1)

        # Берём топ-100 токенов, чтобы не итерироваться по всему словарю
        top_probs, top_indices = torch.topk(probs, k=100)

        result = {}
        for prob, token_id in zip(top_probs.tolist(), top_indices.tolist()):
            token_str = self.tokenizer.decode(token_id)
            result[token_str] = float(prob)

        return result

    def _get_logprobs_batch(self, prompts: list[str]) -> list[dict]:
        """
        Батчевый вариант _get_logprobs.
        Паддит последовательности до одной длины и обрабатывает за один проход.
        """
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,          # паддим до максимальной длины в батче
            truncation=True,
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)

        results = []
        for i, input_ids in enumerate(inputs["input_ids"]):
            # Находим позицию последнего не-паддингового токена
            if inputs.get("attention_mask") is not None:
                last_pos = inputs["attention_mask"][i].sum().item() - 1
            else:
                last_pos = len(input_ids) - 1

            next_token_logits = outputs.logits[i, last_pos, :]
            probs = torch.softmax(next_token_logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=100)

            result = {}
            for prob, token_id in zip(top_probs.tolist(), top_indices.tolist()):
                token_str = self.tokenizer.decode(token_id)
                result[token_str] = float(prob)

            results.append(result)

        return results

    def _aggregate_0_100_score(self, score: dict) -> float | None:
        """
        Взвешенное среднее по токенам 0–100.
        Если суммарная вероятность числовых токенов < 0.25 — возвращает None.
        """
        total = 0.0
        weighted_sum = 0.0

        for token, prob in score.items():
            try:
                int_key = int(token.strip())
            except ValueError:
                continue
            if not (0 <= int_key <= 100):
                continue
            weighted_sum += int_key * prob
            total += prob

        if total < 0.25:
            return None

        return 20 * weighted_sum / total

    def judge(self, metric, **kwargs) -> float | None:
        prompt = self._build_prompt(metric, **kwargs)
        logprobs = self._get_logprobs(prompt)
        return self._aggregate_0_100_score(logprobs)

    def judge_batch(self, metric, batch: list[dict]) -> list[float | None]:
        """
        Батчевый вызов — эффективнее при большом количестве примеров.
        """
        prompts = [self._build_prompt(metric, **kwargs) for kwargs in batch]
        all_logprobs = self._get_logprobs_batch(prompts)
        return [self._aggregate_0_100_score(lp) for lp in all_logprobs]

    def __call__(self, metric, **kwargs) -> float | None:
        return self.judge(metric, **kwargs)
