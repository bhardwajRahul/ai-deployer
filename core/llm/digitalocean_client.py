from typing import Optional
import datetime

import tiktoken
from httpx import Timeout
from openai import AsyncOpenAI, RateLimitError

from core.config import LLMProvider
from core.llm.base import BaseLLMClient
from core.llm.convo import Convo
from core.log import get_logger

log = get_logger(__name__)
tokenizer = tiktoken.get_encoding("cl100k_base")


class DoAIClient(BaseLLMClient):
    provider = LLMProvider.DIGITALOCEAN
    stream_options = {"include_usage": True}

    def _init_client(self):
        self.client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=Timeout(
                max(self.config.connect_timeout, self.config.read_timeout),
                connect=self.config.connect_timeout,
                read=self.config.read_timeout,
            ),
        )

    async def _make_request(
        self,
        convo: Convo,
        temperature: Optional[float] = None,
        json_mode: bool = False,
    ) -> tuple[str, int, int]:
        completion_kwargs = {
            "model": self.config.model,
            "messages": convo.messages,
            "temperature": self.config.temperature if temperature is None else temperature,
            "stream": True,
        }
        if self.stream_options:
            completion_kwargs["stream_options"] = self.stream_options

        if json_mode:
            completion_kwargs["response_format"] = {"type": "json_object"}

        stream = await self.client.chat.completions.create(**completion_kwargs)
        response = []
        prompt_tokens = 0
        completion_tokens = 0

        async for chunk in stream:
            if chunk.usage:
                prompt_tokens += chunk.usage.prompt_tokens
                completion_tokens += chunk.usage.completion_tokens

            if not chunk.choices:
                continue

            content = chunk.choices[0].delta.content
            if not content:
                continue

            response.append(content)
            if self.stream_handler:
                await self.stream_handler(content)

        response_str = "".join(response)

        # Tell the stream handler we're done
        if self.stream_handler:
            await self.stream_handler(None)

        if prompt_tokens == 0 and completion_tokens == 0:
            # See https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
            prompt_tokens = sum(3 + len(tokenizer.encode(msg["content"])) for msg in convo.messages)
            completion_tokens = len(tokenizer.encode(response_str))
            log.warning(
                "DO GenAI response did not include token counts, estimating with tiktoken: "
                f"{prompt_tokens} input tokens, {completion_tokens} output tokens"
            )

        return response_str, prompt_tokens, completion_tokens
    
def rate_limit_sleep(self, err: RateLimitError) -> Optional[datetime.timedelta]:
    """
    Handle rate-limiting by calculating the appropriate wait time based on headers.

    :param err: The RateLimitError raised by the client.
    :return: A timedelta indicating how long to wait before retrying, or None if no wait is needed.
    """
    headers = err.response.headers
    if not headers:
        return None

    # Extract rate-limiting headers
    remaining_requests = int(headers.get("x-ratelimit-remaining-requests", 0))
    remaining_tokens_per_minute = int(headers.get("x-ratelimit-remaining-tokens-per-minute", 0))
    reset_requests = int(headers.get("x-ratelimit-reset-requests", 0))
    reset_tokens_per_minute = int(headers.get("x-ratelimit-reset-tokens-per-minute", 0))

    # Determine wait time based on reset values
    if remaining_requests == 0 and reset_requests > 0:
        wait_time = reset_requests - int(datetime.datetime.now().timestamp())
        log.warning(f"Rate limit hit for requests. Waiting for {wait_time} seconds.")
        return datetime.timedelta(seconds=max(wait_time, 1))  # Ensure at least 1 second wait

    if remaining_tokens_per_minute == 0 and reset_tokens_per_minute > 0:
        wait_time = reset_tokens_per_minute - int(datetime.datetime.now().timestamp())
        log.warning(f"Rate limit hit for tokens per minute. Waiting for {wait_time} seconds.")
        return datetime.timedelta(seconds=max(wait_time, 1))  # Ensure at least 1 second wait

    # If no reset time is provided but limits are exceeded, use a default wait time
    if remaining_requests == 0 or remaining_tokens_per_minute == 0:
        log.warning("Rate limit hit but no reset time provided. Defaulting to 60 seconds.")
        return datetime.timedelta(seconds=60)

    # No rate-limiting detected
    return None
