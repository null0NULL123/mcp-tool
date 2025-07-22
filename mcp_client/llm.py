"""
Original source: https://github.com/sooperset/mcp-client-slackbot
License: MIT
"""

import asyncio
from typing import Dict, List

import httpx


class LLMClient:
    """Client for communicating with LLM APIs."""

    def __init__(self, api_key: str, model: str) -> None:
        """Initialize the LLM client.

        Args:
            api_key: API key for the LLM provider
            model: Model identifier to use
        """
        self.api_key = api_key
        self.model = model
        self.timeout = 30.0  # 30 second timeout
        self.max_retries = 2

    async def get_response(self, messages: List[Dict[str, str]]) -> str:
        """Get a response from the LLM.

        Args:
            messages: List of conversation messages

        Returns:
            Text response from the LLM
        """
        if self.model.startswith("gpt-") or self.model.startswith("ft:gpt-"):
            return await self._get_openai_response(messages)
        elif self.model.startswith("llama-"):
            return await self._get_groq_response(messages)
        elif self.model.startswith("claude-"):
            return await self._get_anthropic_response(messages)
        else:
            raise ValueError(f"Unsupported model: {self.model}")

    async def _get_openai_response(self, messages: List[Dict[str, str]]) -> str:
        """Get a response from the OpenAI API."""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1500,
        }

        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload, headers=headers)

                    if response.status_code == 200:
                        response_data = response.json()
                        return response_data["choices"][0]["message"]["content"]
                    else:
                        if attempt == self.max_retries:
                            return (
                                f"Error from API: {response.status_code} - "
                                f"{response.text}"
                            )
                        await asyncio.sleep(2**attempt)  # Exponential backoff
            except Exception as e:
                if attempt == self.max_retries:
                    return f"Failed to get response: {str(e)}"
                await asyncio.sleep(2**attempt)  # Exponential backoff

    async def _get_groq_response(self, messages: List[Dict[str, str]]) -> str:
        """Get a response from the Groq API."""
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1500,
        }

        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload, headers=headers)

                    if response.status_code == 200:
                        response_data = response.json()
                        return response_data["choices"][0]["message"]["content"]
                    else:
                        if attempt == self.max_retries:
                            return (
                                f"Error from API: {response.status_code} - "
                                f"{response.text}"
                            )
                        await asyncio.sleep(2**attempt)  # Exponential backoff
            except Exception as e:
                if attempt == self.max_retries:
                    return f"Failed to get response: {str(e)}"
                await asyncio.sleep(2**attempt)  # Exponential backoff

    async def _get_anthropic_response(self, messages: List[Dict[str, str]]) -> str:
        """Get a response from the Anthropic API."""
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "anthropic-version": "2023-06-01",
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        # Convert messages to Anthropic format
        system_message = None
        anthropic_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            elif msg["role"] == "user":
                anthropic_messages.append(
                    {"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                anthropic_messages.append(
                    {"role": "assistant", "content": msg["content"]}
                )

        payload = {
            "model": self.model,
            "messages": anthropic_messages,
            "temperature": 0.7,
            "max_tokens": 1500,
        }

        if system_message:
            payload["system"] = system_message

        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload, headers=headers)

                    if response.status_code == 200:
                        response_data = response.json()
                        return response_data["content"][0]["text"]
                    else:
                        if attempt == self.max_retries:
                            return (
                                f"Error from API: {response.status_code} - "
                                f"{response.text}"
                            )
                        await asyncio.sleep(2**attempt)  # Exponential backoff
            except Exception as e:
                if attempt == self.max_retries:
                    return f"Failed to get response: {str(e)}"
                await asyncio.sleep(2**attempt)  # Exponential backoff
