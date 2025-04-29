import json
from typing import List, AsyncGenerator, Dict, Optional
from openai import (
    NOT_GIVEN,
    AsyncStream,  # Keep for type hints if needed, but don't instantiate
)
from openai.types.chat import ChatCompletionMessageParam
import httpx

from pipecat.services.openai import OpenAILLMService
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from typing import TypedDict
from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)


# BasetenChunkData definition remains the same
class BasetenChunkData(TypedDict):
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict]
    usage: Optional[Dict]


class BaseTenSGLangService(OpenAILLMService):
    def __init__(
        self,
        *,
        model: str,
        api_key=None,
        baseten_endpoint="https://model-7wln1dv3.api.baseten.co/environments/production/predict",
        base_url=None,
        organization=None,
        project=None,
        params: BaseOpenAILLMService.InputParams = BaseOpenAILLMService.InputParams(),
        **kwargs,
    ):
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            project=project,
            params=params,
            **kwargs,
        )
        self.baseten_endpoint = baseten_endpoint
        self._baseten_api_key = api_key

        self._baseten_httpx_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=100, max_connections=1000, keepalive_expiry=None
            )
        )

    async def get_chat_completions_generator(
        self, context: OpenAILLMContext, messages: List[ChatCompletionMessageParam]
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        logger.debug(f"{self}: Generating chat [{messages}]")

        request_data = {
            "model": self.model_name,
            "stream": True,
            "messages": messages,
            "temperature": self._settings.get("temperature")
            if self._settings.get("temperature") is not NOT_GIVEN
            else None,
            "top_p": self._settings.get("top_p")
            if self._settings.get("top_p") is not NOT_GIVEN
            else None,
            "max_tokens": self._settings.get("max_tokens")
            if self._settings.get("max_tokens") is not NOT_GIVEN
            else None,
        }
        request_data = {k: v for k, v in request_data.items() if v is not None}
        request_data.update(self._settings.get("extra", {}))

        try:
            headers = {"Authorization": f"Api-Key {self._baseten_api_key}"}

            async with self._baseten_httpx_client.stream(
                "POST",
                self.baseten_endpoint,
                headers=headers,
                json=request_data,
                timeout=None,
            ) as response:
                if response.status_code == 200:
                    async for line in response.aiter_lines():
                        if line.startswith("data:"):
                            try:
                                json_str = line[len("data:") :].strip()
                                if json_str == "[DONE]":
                                    break
                                if json_str:
                                    baseten_chunk_data: BasetenChunkData = json.loads(json_str)
                                    try:
                                        yield ChatCompletionChunk(**baseten_chunk_data)
                                    except TypeError as te:
                                        logger.error(
                                            f"Mismatch converting Baseten chunk keys to ChatCompletionChunk: {te}. Data: {baseten_chunk_data}"
                                        )
                                        yield ChatCompletionChunk(
                                            id="error",
                                            object="chat.completion.chunk",
                                            created=0,
                                            model=self.model_name,
                                            choices=[],
                                        )
                                else:
                                    logger.warning("Received empty data line from Baseten stream.")
                            except json.JSONDecodeError as e:
                                logger.error(
                                    f"Error decoding JSON from Baseten stream: {e}, line: {line}"
                                )
                                yield ChatCompletionChunk(
                                    id="error",
                                    object="chat.completion.chunk",
                                    created=0,
                                    model=self.model_name,
                                    choices=[],
                                )
                        elif line.strip():
                            pass
                else:
                    error_text = await response.aread()
                    logger.error(
                        f"Baseten API Error: Status {response.status_code}, Response: {error_text.decode()}"
                    )
                    yield ChatCompletionChunk(
                        id="error",
                        object="chat.completion.chunk",
                        created=0,
                        model=self.model_name,
                        choices=[
                            {
                                "delta": {"content": f"Error: {response.status_code}"},
                                "finish_reason": "error",
                                "index": 0,
                            }
                        ],
                    )

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error connecting to Baseten: {e}")
            yield ChatCompletionChunk(
                id="error",
                object="chat.completion.chunk",
                created=0,
                model=self.model_name,
                choices=[
                    {
                        "delta": {"content": f"HTTP Error: {e.response.status_code}"},
                        "finish_reason": "error",
                        "index": 0,
                    }
                ],
            )
        except Exception as e:
            logger.exception(f"General error during Baseten streaming: {e}", exc_info=True)
            yield ChatCompletionChunk(
                id="error",
                object="chat.completion.chunk",
                created=0,
                model=self.model_name,
                choices=[
                    {"delta": {"content": f"Error: {e}"}, "finish_reason": "error", "index": 0}
                ],
            )

    async def _stream_chat_completions(
        self, context: OpenAILLMContext
    ) -> AsyncStream[ChatCompletionChunk]:
        """
        Overrides the base method to consume the custom Baseten generator
        and return it wrapped in a way that *looks* like the expected AsyncStream.
        """
        messages: List[ChatCompletionMessageParam] = context.get_messages()

        generator = self.get_chat_completions_generator(context, messages)
        return generator
