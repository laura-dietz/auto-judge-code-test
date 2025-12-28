# minimalllm/dspy_lm.py
from __future__ import annotations

import asyncio
import inspect
from typing import Any, Dict, List, Optional, Type

import dspy


from .llm_config import MinimaLlmConfig
from .llm_protocol import MinimaLlmRequest, MinimaLlmResponse
from .minimal_llm import OpenAIMinimaLlm



def _resolve_dspy_base_lm() -> Type[Any]:
    if hasattr(dspy, "BaseLM"):
        return dspy.BaseLM
    for mod_name, attr in [
        ("dspy.clients", "BaseLM"),
        ("dspy.clients.base", "BaseLM"),
        ("dspy.clients.lm", "BaseLM"),
        ("dspy.models", "BaseLM"),
        ("dspy.lm", "BaseLM"),
    ]:
        try:
            mod = __import__(mod_name, fromlist=[attr])
            if hasattr(mod, attr):
                return getattr(mod, attr)
        except Exception:
            pass
    raise RuntimeError("Could not locate DSPy BaseLM")


_BaseLM = _resolve_dspy_base_lm()


class MinimaLlmDSPyLM(_BaseLM):  # type: ignore[misc]
    """
    DSPy BaseLM adapter that routes all DSPy calls through OpenAIMinimaLlm.

    Design goals:
    - No LiteLLM dependency
    - Compatible with DSPy 2.x and 3.x
    - Defensive against future BaseLM signature changes
    """

    def __init__(self, minimallm: OpenAIMinimaLlm, **kwargs: Any):
        self._minimallm = minimallm
        model_value = getattr(minimallm.cfg, "model", "minimalllm")

        # ----------------------------
        # Robust BaseLM initialization
        # ----------------------------
        init_kwargs: Dict[str, Any] = {}
        try:
            sig = inspect.signature(_BaseLM.__init__)  # type: ignore[arg-type]
            params = sig.parameters

            # Required / common parameters across versions
            if "model" in params:
                init_kwargs["model"] = model_value
            elif "model_name" in params:
                init_kwargs["model_name"] = model_value

            # Forward only kwargs that BaseLM explicitly accepts
            for k, v in kwargs.items():
                if k in params:
                    init_kwargs[k] = v

            super().__init__(**init_kwargs)  # type: ignore[misc]

        except Exception:
            # Extremely defensive fallback chain
            try:
                super().__init__(model=model_value)  # type: ignore[misc]
            except Exception:
                try:
                    super().__init__(model_value)  # type: ignore[misc]
                except Exception:
                    super().__init__()  # type: ignore[misc]

        # ----------------------------
        # Commonly expected attributes
        # ----------------------------
        if not hasattr(self, "model"):
            self.model = model_value  # type: ignore[assignment]
        if not hasattr(self, "history"):
            self.history = []  # type: ignore[assignment]
        if not hasattr(self, "kwargs"):
            self.kwargs = {}  # type: ignore[assignment]

    async def acall(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        if messages is None:
            if prompt is None:
                raise ValueError("DSPy LM requires either prompt or messages")
            messages = [{"role": "user", "content": prompt}]

        req = MinimaLlmRequest(
            request_id=str(kwargs.pop("request_id", "dspy")),
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
            temperature=kwargs.pop("temperature", None),
            max_tokens=kwargs.pop("max_tokens", None),
            extra=kwargs,
        )
        resp = await self._minimallm.prompt_one(req)
        return [resp.text]

    async def aforward(self, *args: Any, **kwargs: Any) -> List[str]:
        return await self.acall(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> List[str]:
        return self.__call__(*args, **kwargs)

    def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            raise RuntimeError(
                "MinimaLlmDSPyLM was called synchronously inside a running event loop. "
                "Use await pred.acall(...) or await lm.acall(...)."
            )

        return asyncio.run(self.acall(prompt=prompt, messages=messages, **kwargs))


import dspy

async def example() -> None:
    cfg = MinimaLlmConfig.from_env()
    backend = OpenAIMinimaLlm(cfg)

    lm = MinimaLlmDSPyLM(backend)
    dspy.configure(lm=lm)

    class Sig(dspy.Signature):
        prompt: str = dspy.InputField()
        answer: str = dspy.OutputField()

    pred = dspy.Predict(Sig)
    try:
        out = await pred.acall(prompt=f"Say 'hi'")
        print(out.answer)
    finally:
        await backend.aclose()


if __name__ == "__main__":
    asyncio.run(example())
    # asyncio.run(main())
    # example()
