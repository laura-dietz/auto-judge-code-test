# import asyncio
# from typing import List

from .llm_protocol import MinimaLlmRequest, MinimaLlmResponse
from .minimal_llm import MinimaLlmFailure, OpenAIMinimaLlm
from .llm_config import MinimaLlmConfig


# # --------------------
# # Build a batch of requests
# # --------------------

# requests = [
#     MinimaLlmRequest(
#         request_id=f"q{i}",
#         messages=[
#             {"role": "system", "content": "You are a fair and careful evaluator."},
#             {"role": "user", "content": f"Evaluate answer #{i} for correctness."},
#         ],
#         temperature=0.0,
#     )
#     for i in range(1000)
# ]


# # --------------------
# # Run the batch
# # --------------------

# async def main() -> None:
#     backend = OpenAIMinimaLlm(MinimaLlmConfig.from_env())

#     # Always log the effective configuration for long runs
#     print(backend.cfg.describe())
#     print("-" * 60)

#     try:
#         results = await backend.run_batched(requests)
#     except RuntimeError as e:
#         # Batch-level abort (e.g., too many failures)
#         print(f"Batch aborted: {e}")
#         return

#     # Separate successes from failures
#     ok: List[MinimaLlmResponse] = []
#     failed: List[MinimaLlmFailure] = []

#     for r in results:
#         if isinstance(r, MinimaLlmResponse):
#             ok.append(r)
#         else:
#             failed.append(r)

#     print(f"Completed: {len(ok)}")
#     print(f"Failed:    {len(failed)}")

#     # Example: print first few judgments
#     for r in ok[:3]:
#         print(r.request_id, "→", r.text[:80])


# if __name__ == "__main__":
#     asyncio.run(main())


import asyncio
from typing import List

from .minimal_llm import OpenAIMinimaLlm



class DSPyMinimaLlm(OpenAIMinimaLlm):
    """
    Example backend that uses DSPy to produce completions, while reusing
    MinimaLlm's batching/heartbeat/failure handling via `run_batched()`.

    Participant-side install:
      pip install dspy-ai

    Notes:
    - DSPy APIs vary slightly by version. The `configure_dspy()` function is
      isolated so participants can tweak it easily.
    - This example treats the final user turn as the "prompt". If your eval
      needs full chat history, change `_messages_to_prompt()`.
    """

    def __init__(self, cfg: MinimaLlmConfig):
        super().__init__(cfg)
        try:
            import dspy  # type: ignore
        except ImportError as e:
            raise RuntimeError("Install dspy-ai to use DSPyMinimaLlm") from e

        self._dspy = dspy
        self._configure_dspy()

    @classmethod
    def from_env(cls) -> "DSPyMinimaLlm":
        return cls(MinimaLlmConfig.from_env())

    def _configure_dspy(self) -> None:
        """
        Configure DSPy LM to point at an OpenAI-compatible endpoint.

        This is the only place that is DSPy-version-sensitive.
        Participants can adjust this to match their DSPy release / LM wrapper.
        """
        cfg = self.cfg
        dspy = self._dspy

        # --- Common pattern in DSPy setups ---
        # Depending on DSPy version, this might be:
        #   dspy.LM(...)
        #   dspy.OpenAI(...)
        #   dspy.LiteLLM(...)
        #
        # Keep it here so participants can patch one small block.
        lm = dspy.LM(  # type: ignore[attr-defined]
            model=cfg.model,
            api_base=cfg.base_url,
            api_key=cfg.api_key,  # may be None for local endpoints
            provider="openai",
        )
        dspy.settings.configure(lm=lm)  # type: ignore[attr-defined]

    @staticmethod
    def _messages_to_prompt(messages) -> str:
        # Minimal default: concatenate user turns.
        return "\n".join(m["content"] for m in messages if m.get("role") == "user")

    async def prompt_one(self, req: MinimaLlmRequest) -> MinimaLlmResponse:
        dspy = self._dspy
        prompt = self._messages_to_prompt(req.messages)

        class Sig(dspy.Signature):  # type: ignore[attr-defined]
            prompt: str = dspy.InputField()   # type: ignore[attr-defined]
            answer: str = dspy.OutputField()  # type: ignore[attr-defined]

        predictor = dspy.Predict(Sig)  # type: ignore[attr-defined]
        res = await predictor.acall(prompt=prompt)

        return MinimaLlmResponse(
            request_id=req.request_id,
            text=str(res.answer),
            raw=None,
        )


# --------------------
# Build a batch of requests
# --------------------

requests = [
    MinimaLlmRequest(
        request_id=f"q{i}",
        messages=[
            {"role": "system", "content": "You are a fair and careful evaluator."},
            {"role": "user", "content": f"Evaluate answer #{i} for correctness."},
        ],
        temperature=0.0,
    )
    for i in range(1000)
]


# --------------------
# Run the batch (same harness behavior as OpenAIMinimaLlm)
# --------------------

async def main() -> None:
    cfg = MinimaLlmConfig.from_env()
    backend = OpenAIMinimaLlm(cfg)

    lm = MinimaLlmDSPyLM(backend)
    dspy.configure(lm=lm)


    print(backend.cfg.describe())
    print("-" * 60)

    try:
        results = await backend.run_batched(requests)
    except RuntimeError as e:
        print(f"Batch aborted: {e}")
        return

    ok: List[MinimaLlmResponse] = []
    failed: List[MinimaLlmFailure] = []

    for r in results:
        if isinstance(r, MinimaLlmResponse):
            ok.append(r)
        else:
            failed.append(r)

    print(f"Completed: {len(ok)}")
    print(f"Failed:    {len(failed)}")

    for r in ok[:3]:
        print(r.request_id, "→", r.text[:80])


if __name__ == "__main__":
    asyncio.run(main())
