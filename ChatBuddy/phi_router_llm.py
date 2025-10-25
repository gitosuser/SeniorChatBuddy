# phi_router_llm.py

from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class PhiRouterLLM:
    """
    Tiny local "router brain" that decides which high-level action
    the chat buddy should take next, given conversation state.

    It MUST return exactly one of:
      - FETCH_WEATHER
      - ADVISE_FROM_WEATHER
      - FETCH_DIRECTORY
      - CHAT
    """

    def __init__(
        self,
        model_id: str = "microsoft/phi-3.5-mini-instruct",
        max_new_tokens: int = 8,
    ):
        self.max_new_tokens = max_new_tokens

        print(f"[PhiRouterLLM] Loading {model_id} with transformers (CPU)...")

        # Force pure CPU path
        self.device = torch.device("cpu")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Try bfloat16 first (Apple Silicon supports bfloat16 math in torch 2.9),
        # fall back to full float32 if it complains.
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.bfloat16,   # <-- NOTE: dtype=..., not torch_dtype=...
            )
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.float32,
            )

        # Manually move model to CPU. We are NOT passing device_map anywhere.
        self.model.to(self.device)
        self.model.eval()

        print("[PhiRouterLLM] Router model ready (transformers/CPU).")

    def _build_router_prompt(
        self,
        user_input: str,
        last_weather_summary: Optional[str],
        recent_chat_turns: str,
    ) -> str:
        summary_block = last_weather_summary or "None"

        # IMPORTANT: we explicitly forbid any extra text.
        prompt = f"""You are a routing classifier for a senior-friendly chat assistant.
Your job is to choose exactly ONE next action label, with no explanation.

Conversation so far:
{recent_chat_turns}

User just said:
\"{user_input}\"

Most recent weather summary (if any):
{summary_block}

Choose ONE next action:

FETCH_WEATHER
- User is asking about outside weather, rain, cold/warm, what to wear, or if it's okay to go for a walk AND we have NOT already answered weather this exact turn.

ADVISE_FROM_WEATHER
- The user is asking follow-up safety/comfort questions (like “Is it safe to walk?” or “Do I need a jacket?”) AND we already have a recent weather summary.

FETCH_DIRECTORY
- The user is asking for a phone number, address, location, hours, or similar about a store, clinic, pharmacy, or other local service.

CHAT
- Anything else: personal stories, feelings, emotional support, general talk.

You MUST answer with ONLY one of these tokens:
FETCH_WEATHER
ADVISE_FROM_WEATHER
FETCH_DIRECTORY
CHAT
Do not include any other words or punctuation.
"""
        return prompt




    def invoke(
        self,
        user_input: str,
        last_weather_summary: Optional[str],
        recent_chat_turns: str,
    ):
        """
        Run the small model and return a tiny object with .content
        holding the chosen action token.
        """

        prompt = self._build_router_prompt(
            user_input=user_input,
            last_weather_summary=last_weather_summary,
            recent_chat_turns=recent_chat_turns,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Take only newly generated part
        gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        # First word is our routing label
        action_token = raw_text.strip().split()[0] if raw_text.strip() else ""

        class Result:
            def __init__(self, content: str):
                self.content = content

        return Result(action_token)

