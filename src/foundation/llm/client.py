from google import genai
from google.genai import types

from foundation.config import settings
from foundation.llm.schemas import ChatResponse


class GeminiClient:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key if api_key is not None else settings.google_api_key
        self.model = model or settings.gemini_model
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY missing")
        self._client = genai.Client(api_key=self.api_key)

    def chat(self, question: str) -> ChatResponse:
        response = self._client.models.generate_content(
            model=self.model,
            contents=question,
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
                response_json_schema=ChatResponse.model_json_schema(),
            ),
        )
        if getattr(response, "parsed", None) is not None:
            parsed = response.parsed
            if isinstance(parsed, ChatResponse):
                return parsed
        if not response.text:
            raise ValueError("Empty Gemini response")
        return ChatResponse.model_validate_json(response.text)
