# minima_llm_dspy.py
from __future__ import annotations

import typing
from typing import Any, Callable, Dict, List, Optional, Type, Union, cast, get_args

import asyncio
import contextvars
import inspect
import os
import re


try:
    import dspy
    from dspy.adapters.chat_adapter import ChatAdapter
except ImportError as e:
    raise ImportError(
        "This module requires DSPy. Install with: pip install trec_auto_judge[dspy]"
    ) from e

def _import_adapter_parse_error():
    """Locate AdapterParseError across DSPy versions.
    0.1.3 - 0.1.6 (pre-refactor)
    0.1.7 -  0.1.9 (adapter split)
    early 0.2.x nightlies (exceptions module split)
    """
    paths = [
        "dspy.adapters.exceptions",
        "dspy.adapters.base",
        "dspy.adapters",
        "dspy.primitives.exceptions",
        "dspy.exceptions",
        "dspy.utils.exceptions",
    ]

    for path in paths:
        try:
            module = __import__(path, fromlist=["AdapterParseError"])
            return module.AdapterParseError
        except (ImportError, AttributeError):
            continue

    # Fallback: define compatible exception for older DSPy versions
    dspy_version = getattr(dspy, "__version__", "unknown")
    print(f"Warning: AdapterParseError not found in DSPy {dspy_version}, using fallback class")

    class AdapterParseError(Exception):
        """Fallback AdapterParseError for DSPy versions without this exception."""
        def __init__(self, adapter_name=None, signature=None, lm_response=None,
                     parsed_result=None, message=None, **kwargs):
            self.adapter_name = adapter_name
            self.signature = signature
            self.lm_response = lm_response
            self.parsed_result = parsed_result
            super().__init__(message or "Adapter parse error")

    return AdapterParseError


AdapterParseError = _import_adapter_parse_error()

from pydantic import BaseModel

from .llm_protocol import MinimaLlmRequest
from .minima_llm import MinimaLlmFailure, OpenAIMinimaLlm, get_force_refresh, reset_force_refresh, set_force_refresh, set_last_cached


# ====== More tolerant chat adapter ========


class TolerantChatAdapter(ChatAdapter):
    # Matches a well-formed header anywhere in a line, e.g. [[ ## answerability ## ]]
    # Group captures the raw field name between the ## ... ## markers.
    _HEADER_RE = re.compile(
        r"\[\[\s*##\s*(?P<name>[^#\]\r\n]+?)\s*##\s*\]\]",
        flags=re.IGNORECASE,
    )

    @classmethod
    def normalize_field_name(cls, raw: str) -> str:
        # Mirror your old normalization: lower + spaces to underscores
        return raw.strip().lower().replace(" ", "_")

    @classmethod
    def is_optional_type(cls, tp):
        """Return True if annotation is Optional[...]."""
        return (
            getattr(tp, "__origin__", None) is typing.Union
            and type(None) in getattr(tp, "__args__", ())
        )

    @classmethod
    def unwrap_optional(cls, ann) -> type:
        """Unwrap Optional[T] to T, or return ann unchanged."""
        if getattr(ann, "__origin__", None) is typing.Union:
            args = getattr(ann, "__args__", ())
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return non_none[0]
        return ann

    @classmethod
    def is_type(cls, ann, target: type) -> bool:
        """Return True if annotation is target or Optional[target]."""
        return cls.unwrap_optional(ann) is target

    # Convenience methods for common types
    is_float = classmethod(lambda cls, ann: cls.is_type(ann, float))
    is_int = classmethod(lambda cls, ann: cls.is_type(ann, int))

    # Regex patterns for numeric parsing
    _FLOAT_PATTERN = re.compile(r"[-+]?[0-9]*\.?[0-9]+")
    _INT_PATTERN = re.compile(r"[-+]?\d+")

    @classmethod
    def try_parse_numeric(cls, val, target: type, pattern: re.Pattern) -> int | float:
        """Parse numeric value from LLM output. Raises ValueError on failure."""
        try:
            m = pattern.search(str(val).strip())
            if m:
                return target(m.group())
        except (ValueError, TypeError):
            pass
        raise ValueError(f"Could not parse {target.__name__}: {str(val)[:50]}")

    @classmethod
    def is_list_str(cls, ann):
        """Return True if annotation is list[str], List[str], or Optional[list[str]]."""
        from typing import Union
        origin = getattr(ann, "__origin__", None)

        # Handle Optional[List[str]] -> Union[List[str], None]
        if origin is Union:
            args = getattr(ann, "__args__", ())
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return cls.is_list_str(non_none[0])
            return False

        if origin is list:
            args = getattr(ann, "__args__", ())
            return args == (str,)
        return False

    # Regex patterns for list format detection
    _SINGLE_QUOTE_LIST = re.compile(r"^\s*\[\s*'")
    _DOUBLE_QUOTE_LIST = re.compile(r'^\s*\[\s*"')
    _UNESCAPED_SINGLE = re.compile(r"(?<!\\)'")

    @classmethod
    def try_parse_list_str(cls, val: str) -> list[str]:
        """Parse list from JSON or Python syntax.

        - Double-quote lists (JSON): parse directly, handles apostrophes naturally
        - Single-quote lists (Python): validate balanced quotes, parse if valid
        - On failure: raise ValueError to trigger retry with fresh LLM call
        """
        import ast
        import json

        val = val.strip()

        if cls._DOUBLE_QUOTE_LIST.match(val):
            # JSON-style double quotes - preferred format, apostrophes are safe
            try:
                parsed = json.loads(val)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if x]
            except json.JSONDecodeError:
                pass
            raise ValueError(f"Invalid JSON array: {val[:100]}")

        elif cls._SINGLE_QUOTE_LIST.match(val):
            # Python-style single quotes - check for balanced quotes first
            quote_count = len(cls._UNESCAPED_SINGLE.findall(val))
            if quote_count % 2 != 0:
                raise ValueError(
                    f"Unbalanced single quotes ({quote_count}) - "
                    f"likely apostrophe issue, use JSON format: {val[:100]}"
                )

            # Even quotes - attempt parse (will fail if apostrophes cause issues)
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if x]
            except (ValueError, SyntaxError) as e:
                raise ValueError(f"Invalid Python list ({e}), use JSON format: {val[:100]}")

        # Empty list [] or other format - try JSON
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if x]
        except json.JSONDecodeError:
            pass

        raise ValueError(f"Expected JSON array, got: {val[:100]}")

    @classmethod
    def _is_non_value(cls, s: str) -> bool:
        return s.strip().lower() in {"", "none", "null"}

    # -------------------------------------------------------------------------
    # Parse pipeline: extract → reduce → coerce
    # -------------------------------------------------------------------------

    def _extract_sections(self, completion: str) -> list[tuple[str | None, list[str]]]:
        """Extract (header, lines) sections from completion using [[ ## field ## ]] markers."""
        sections: list[tuple[str | None, list[str]]] = [(None, [])]
        current_lines = sections[-1][1]

        for raw_line in completion.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            last_end = 0
            for m in self._HEADER_RE.finditer(line):
                # Text before header → current section
                before = line[last_end:m.start()].strip()
                if before:
                    current_lines.append(before)

                # Start new section at this header
                header = self.normalize_field_name(m.group("name"))
                sections.append((header, []))
                current_lines = sections[-1][1]
                last_end = m.end()

            # Trailing text → current section
            after = line[last_end:].strip()
            if after:
                current_lines.append(after)

        return sections

    def _sections_to_dict(self, sections: list[tuple[str | None, list[str]]], output_fields: set[str]) -> dict[str, str]:
        """Reduce sections to {field: value} for known output fields."""
        parsed: dict[str, str] = {}
        for key, lines in sections:
            if key in output_fields:
                val = "\n".join(lines).strip()
                if not self._is_non_value(val):
                    parsed[key] = val  # last occurrence wins
        return parsed

    def _coerce_field(self, val: str, annotation: type) -> typing.Any:
        """Coerce string value to annotation type. Raises ValueError on failure."""
        if self.is_float(annotation):
            return float(self.try_parse_numeric(val, float, self._FLOAT_PATTERN))
        if self.is_int(annotation):
            return int(self.try_parse_numeric(val, int, self._INT_PATTERN))
        if self.is_list_str(annotation) and isinstance(val, str):
            return list(self.try_parse_list_str(val))
        return val  # No coercion needed

    def _validate_and_coerce(self, parsed: dict[str, str], signature, completion: str) -> dict[str, typing.Any]:
        """Validate required fields present and coerce all values to target types."""
        result: dict[str, typing.Any] = {}

        for name, field in signature.output_fields.items():
            annotation = field.annotation

            if name in parsed:
                try:
                    result[name] = self._coerce_field(parsed[name], annotation)
                except (ValueError, TypeError) as e:
                    raise AdapterParseError(
                        adapter_name="TolerantChatAdapter",
                        signature=signature,
                        lm_response=completion,
                        parsed_result=parsed,
                        message=str(e),
                    )
            elif self.is_optional_type(annotation):
                result[name] = None
            else:
                raise AdapterParseError(
                    adapter_name="TolerantChatAdapter",
                    signature=signature,
                    lm_response=completion,
                    parsed_result=parsed,
                    message=f"Missing required field: {name}",
                )

        return result

    def parse(self, signature, completion: str) -> dict[str, typing.Any]:
        """Parse LLM completion into typed output fields.

        Pipeline: extract sections → reduce to dict → validate & coerce types
        """
        sections = self._extract_sections(completion)
        parsed = self._sections_to_dict(sections, set(signature.output_fields.keys()))
        return self._validate_and_coerce(parsed, signature, completion)

    
def _get_dspy_version() -> tuple[int, int, int]:
    """Parse DSPy version into (major, minor, patch) tuple."""
    version_str = getattr(dspy, "__version__", "0.0.0")
    try:
        parts = version_str.split(".")[:3]
        return tuple(int(p) for p in parts)  # type: ignore
    except (ValueError, AttributeError):
        return (0, 0, 0)


def _select_adapter():
    """Select adapter for DSPy.

    Always use TolerantChatAdapter for better list[str] and Optional field handling.
    """
    
    
    # """
    # - DSPy 3.0+: Use stock ChatAdapter with JSON fallback
    # - DSPy < 3.0: Use TolerantChatAdapter for parsing workarounds
    # """
    # version = _get_dspy_version()

    # if version >= (3, 0, 0):
    #     # DSPy 3.x has improved parsing with automatic JSON fallback
    #     return ChatAdapter()
    # else:
    #     # Older DSPy needs our tolerant parsing for list/float handling
    #     return TolerantChatAdapter()

    return TolerantChatAdapter()


_dspy_version = _get_dspy_version()
_adapter = _select_adapter()
print(f"DSPy {'.'.join(map(str, _dspy_version))} loaded, using {type(_adapter).__name__}")
dspy.settings.configure(adapter=_adapter)



# ==============

def _resolve_dspy_base_lm() -> Type[Any]:
    """
    Locate DSPy's BaseLM class across common DSPy layouts.

    DSPy moves internals occasionally; this helper keeps the adapter resilient.
    """
    if hasattr(dspy, "BaseLM"):
        return dspy.BaseLM  # type: ignore[attr-defined]

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
    DSPy BaseLM adapter that routes calls through OpenAIMinimaLlm.

    This adapter is intentionally minimal:
      - DSPy handles prompt construction and output parsing.
      - MinimaLlm handles HTTP transport, backpressure, retries, and pacing.
      - No LiteLLM dependency.
    """

    def __init__(self, minimallm: OpenAIMinimaLlm, **kwargs: Any):
        self._minimallm = minimallm
        model_value = minimallm.cfg.model

        # Initialize BaseLM in a version-tolerant way (DSPy 2.6.27 requires `model`).
        try:
            sig = inspect.signature(_BaseLM.__init__)  # type: ignore[arg-type]
            params = sig.parameters
            init_kwargs: Dict[str, Any] = {}

            if "model" in params:
                init_kwargs["model"] = model_value
            elif "model_name" in params:
                init_kwargs["model_name"] = model_value

            # Forward only kwargs that BaseLM actually accepts.
            for k, v in kwargs.items():
                if k in params:
                    init_kwargs[k] = v

            super().__init__(**init_kwargs)  # type: ignore[misc]
        except Exception:
            # Fallback chain
            try:
                super().__init__(model=model_value)  # type: ignore[misc]
            except Exception:
                try:
                    super().__init__(model_value)  # type: ignore[misc]
                except Exception:
                    super().__init__()  # type: ignore[misc]

        # Commonly expected attributes (harmless if unused)
        if not hasattr(self, "model"):
            self.model = model_value  # type: ignore[assignment]
        if not hasattr(self, "kwargs"):
            self.kwargs = {}  # type: ignore[assignment]
        if not hasattr(self, "history"):
            self.history = []  # type: ignore[assignment]

    async def acall(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        *,
        force_refresh: bool = False,
        **kwargs: Any,
    ) -> List[str]:
        """
        Async LM call used by DSPy.

        Parameters
        ----------
        prompt : str, optional
            Single prompt string (converted to user message)
        messages : list, optional
            OpenAI-format message list
        force_refresh : bool
            If True, bypass cache lookup and make a fresh LLM call.
            Useful for retrying when DSPy parsing fails on a cached response.

        Returns
        -------
        list[str]
            DSPy expects a list of completions. We return a singleton list.
        """
        # Check contextvar for force_refresh (set by retry logic)
        force_refresh = force_refresh or get_force_refresh()

        if messages is None:
            if prompt is None:
                raise ValueError("DSPy LM requires either prompt or messages")
            messages = [{"role": "user", "content": prompt}]

        # Debug: show what kwargs DSPy passes (set MINIMA_DEBUG=1 to enable)
        if os.environ.get("MINIMA_DEBUG") and kwargs:
            print(f"[MinimaLlmDSPyLM] DSPy kwargs: {list(kwargs.keys())}")
            if "response_format" in kwargs:
                print(f"[MinimaLlmDSPyLM] response_format: {kwargs['response_format']}")

        req = MinimaLlmRequest(
            request_id=str(kwargs.pop("request_id", "dspy")),
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
            temperature=kwargs.pop("temperature", None),
            max_tokens=kwargs.pop("max_tokens", None),
            extra=kwargs if kwargs else None,
        )


        resp = await self._minimallm.generate(req, force_refresh=force_refresh)
        if isinstance(resp, MinimaLlmFailure):
            error_msg = f"{resp.error_type}: {resp.message}"
            if resp.body_snippet:
                error_msg += f"\nResponse body: {resp.body_snippet}"
            raise RuntimeError(error_msg)
        set_last_cached(resp.cached)
        return [resp.text]

    # Some DSPy internals/adapters call forward/aforward.
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
        """
        Sync LM call fallback.

        If called inside a running event loop, raise a clear error rather than
        nesting event loops.
        """
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


# ----------------------------
# Batch execution helper
# ----------------------------

def _get_input_field_names(signature_class: Type[dspy.Signature]) -> List[str]:
    """
    Extract InputField names from a DSPy Signature class.

    Returns list of field names that are InputFields in the signature.
    """
    input_fields = []

    # Method 1: Check DSPy's signature-level field collections
    # DSPy stores fields in various places depending on version
    for attr_name in ['input_fields', '_input_fields', 'fields']:
        if hasattr(signature_class, attr_name):
            fields_obj = getattr(signature_class, attr_name)

            # Could be a dict
            if isinstance(fields_obj, dict):
                # Might be {name: field_obj} or nested structure
                for key, value in fields_obj.items():
                    if isinstance(key, str):
                        input_fields.append(key)
                if input_fields:
                    break
            # Could be a dict-like object
            elif hasattr(fields_obj, 'keys') and callable(fields_obj.keys):
                try:
                    input_fields = list(fields_obj.keys())
                    if input_fields:
                        break
                except Exception:
                    pass
            # Could be a list/sequence of field names
            elif hasattr(fields_obj, '__iter__'):
                try:
                    input_fields = [f for f in fields_obj if isinstance(f, str)]
                    if input_fields:
                        break
                except Exception:
                    pass

    if input_fields:
        return input_fields

    # Method 2: Check Pydantic v2 model_fields
    if hasattr(signature_class, 'model_fields'):
        fields_dict = signature_class.model_fields
        for name, field_info in fields_dict.items():
            # Check metadata list (Pydantic v2 standard)
            if hasattr(field_info, 'metadata') and field_info.metadata:
                for meta_item in field_info.metadata:
                    # DSPy might store InputField instance in metadata
                    meta_type = type(meta_item).__name__
                    meta_module = type(meta_item).__module__ if hasattr(type(meta_item), '__module__') else ''
                    if 'InputField' in meta_type or ('dspy' in meta_module and 'Input' in meta_type):
                        input_fields.append(name)
                        break

            # Check json_schema_extra for DSPy markers
            if name not in input_fields and hasattr(field_info, 'json_schema_extra') and field_info.json_schema_extra:
                extra = field_info.json_schema_extra
                if extra.get('__dspy_field_type') == 'input':
                    input_fields.append(name)
                elif extra.get('prefix', '').lower().startswith('input'):
                    input_fields.append(name)

    if input_fields:
        return input_fields

    # Method 3: Check Pydantic v1 __fields__
    if hasattr(signature_class, '__fields__'):
        fields_dict = signature_class.__fields__
        for name, field_info in fields_dict.items():
            if hasattr(field_info, 'field_info'):
                field_info = field_info.field_info
            # Check extra for DSPy markers
            if hasattr(field_info, 'extra') and field_info.extra:
                extra = field_info.extra
                if extra.get('__dspy_field_type') == 'input':
                    input_fields.append(name)
            # Check json_schema_extra
            if hasattr(field_info, 'json_schema_extra') and field_info.json_schema_extra:
                extra = field_info.json_schema_extra
                if extra.get('__dspy_field_type') == 'input':
                    input_fields.append(name)

    if input_fields:
        return input_fields

    # Method 4: Introspect class attributes for Field objects
    for name in signature_class.__annotations__:
        try:
            # Try class attribute
            field_obj = getattr(signature_class, name, None)
            if field_obj is None:
                # Try getting from __dict__
                field_obj = signature_class.__dict__.get(name, None)

            if field_obj is None:
                continue

            # Check type and class name
            field_type_str = str(type(field_obj))
            field_class_name = field_obj.__class__.__name__ if hasattr(field_obj, '__class__') else ''
            field_module = field_obj.__class__.__module__ if hasattr(field_obj, '__class__') else ''

            # Check multiple possible indicators of InputField
            is_input = any([
                'InputField' in field_class_name,
                'InputField' in field_type_str,
                'Input' in field_class_name and 'dspy' in field_module,
                hasattr(field_obj, 'json_schema_extra') and
                    isinstance(field_obj.json_schema_extra, dict) and
                    field_obj.json_schema_extra.get('__dspy_field_type') == 'input',
            ])

            if is_input:
                input_fields.append(name)

        except Exception:
            continue

    if input_fields:
        return input_fields

    # Method 5: Fallback heuristic - fields before first OutputField
    # DSPy signatures conventionally list inputs before outputs
    print(f"Warning: Could not detect InputFields via metadata, using annotation order heuristic")

    output_field_names = []

    # Try to find output fields
    for name in signature_class.__annotations__:
        try:
            field_obj = getattr(signature_class, name, None)
            if field_obj is None:
                field_obj = signature_class.__dict__.get(name, None)

            if field_obj is not None:
                field_type_str = str(type(field_obj))
                field_class_name = field_obj.__class__.__name__ if hasattr(field_obj, '__class__') else ''

                if 'OutputField' in field_class_name or 'OutputField' in field_type_str:
                    output_field_names.append(name)
        except Exception:
            continue

    # If we found output fields, everything before the first output is input
    if output_field_names:
        first_output = output_field_names[0]
        for name in signature_class.__annotations__:
            if name == first_output:
                break
            input_fields.append(name)
        return input_fields

    # Last resort: check if signature has exactly the expected fields from Umbrela
    # Return first 4 annotations as inputs (Umbrela has 4 inputs, 5 outputs)
    annotations = list(signature_class.__annotations__.keys())
    if len(annotations) >= 4:
        # Assume first 4 are inputs
        print(f"Warning: Using first 4 annotations as inputs: {annotations[:4]}")
        return annotations[:4]

    # Give up - return empty to trigger clear error
    print(f"Error: Could not detect any InputFields in signature {signature_class.__name__}")
    return []


async def run_dspy_batch(
    signature_class: Type[dspy.Signature],
    annotation_objs: List[BaseModel],
    output_converter: Callable[[Any, BaseModel], None],
    backend: OpenAIMinimaLlm,
    predictor_class: Type = dspy.ChainOfThought,
) -> List[BaseModel]:
    """
    Execute a DSPy batch with MinimaLLM backend.

    This helper automatically extracts input fields from annotation objects based on
    the DSPy signature's InputFields, executes predictions in parallel using batching,
    and updates annotation objects with results.

    Parameters
    ----------
    signature_class : Type[dspy.Signature]
        DSPy Signature class (e.g., Umbrela)
    annotation_objs : List[BaseModel]
        List of Pydantic models with fields matching signature InputFields
    output_converter : Callable[[Any, BaseModel], None]
        Function that updates annotation object with DSPy prediction result.
        Signature: (prediction: dspy.Prediction, obj: BaseModel) -> None
    predictor_class : Type
        DSPy predictor class (default: dspy.ChainOfThought)
    backend : Optional[OpenAIMinimaLlm]
        Pre-configured backend. If None, creates from environment.

    Returns
    -------
    List[BaseModel]
        Processed annotation objects with outputs filled in

    Example
    -------
    >>> class MyAnnotation(BaseModel):
    ...     title: str
    ...     text: str
    ...     score: Optional[float] = None
    >>>
    >>> class MySignature(dspy.Signature):
    ...     title: str = dspy.InputField()
    ...     text: str = dspy.InputField()
    ...     score: float = dspy.OutputField()
    ...
    ...     @classmethod
    ...     def convert_output(cls, pred, obj):
    ...         obj.score = float(pred.score)
    >>>
    >>> annotations = [MyAnnotation(title="...", text="..."), ...]
    >>> results = await run_dspy_batch(MySignature, annotations, MySignature.convert_output)
    """
    # Setup backend
    owns_backend = backend is None
    if backend is None:
        backend = OpenAIMinimaLlm.from_env()

    lm = MinimaLlmDSPyLM(backend)

    # Get input field names from signature
    input_fields = _get_input_field_names(signature_class)

    # Code errors that should propagate immediately (not retry)
    CODE_ERRORS = (NameError, TypeError, AttributeError, SyntaxError, ImportError)

    # Parse retry limit (separate from HTTP retries in generate())
    http_max_attempts = backend.cfg.max_attempts
    parse_retry_limit = 3 if http_max_attempts == 0 else http_max_attempts

    # Use dspy.context() to support multiple asyncio.run() calls (unlike dspy.settings.configure)
    with dspy.context(lm=lm, adapter=_select_adapter()):
        predictor = predictor_class(signature_class)

        async def _maybe_await(result):
            """Await if awaitable, otherwise return value directly."""
            if inspect.isawaitable(result):
                return await result
            return result

        async def invoke_predictor(pred, **kw):
            """Invoke predictor with version-tolerant async/sync handling."""
            import functools

            for method_name in ("acall", "aforward"):
                method = getattr(pred, method_name, None)
                if callable(method):
                    return await _maybe_await(method(**kw))

            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, functools.partial(pred, **kw))

        async def process_one(obj: BaseModel) -> BaseModel:
            kw = obj.model_dump(include=set(input_fields))
            last_error: Optional[Exception] = None

            for attempt in range(parse_retry_limit):
                force_refresh_token: Optional[contextvars.Token[bool]] = None
                try:
                    if attempt > 0:
                        force_refresh_token = set_force_refresh(True)

                    result = await invoke_predictor(predictor, **kw)
                    output_converter(result, obj)
                    return obj

                except CODE_ERRORS:
                    raise
                except AdapterParseError as e:
                    last_error = e
                    continue
                except Exception as e:
                    last_error = e
                    continue
                finally:
                    if force_refresh_token is not None:
                        reset_force_refresh(force_refresh_token)

            raise last_error  # type: ignore[misc]

        results = await backend.run_batched_callable(annotation_objs, process_one)

    # Check for failures - fail fast, don't silently drop data
    failures = [r for r in results if isinstance(r, MinimaLlmFailure)]
    if failures:
        # Cleanup before raising
        if owns_backend:
            await backend.aclose()
        # Report failures
        msgs = [f"{f.request_id}: {f.error_type}: {f.message}" for f in failures[:5]]
        raise RuntimeError(
            f"{len(failures)} DSPy predictions failed:\n  " + "\n  ".join(msgs)
        )

    # Cleanup
    if owns_backend:
        await backend.aclose()

    # All results are BaseModel (failures already raised)
    return cast(List[BaseModel], results)




T = typing.TypeVar("T", bound=BaseModel)


def run_dspy_batch_generic(
    data: List[T],
    signature: Type[dspy.Signature],
    converter: Callable[[dspy.Prediction, T], None],
    llm_config: "MinimaLlmConfig",
) -> List[T]:
    """
    Run DSPy batch for any data model and signature.

    Convenience wrapper around run_dspy_batch that handles asyncio
    and backend setup.

    Args:
        data: List of Pydantic models to process
        signature: DSPy signature class (required)
        converter: Output converter function to populate data from prediction
        llm_config: LLM configuration

    Returns:
        Updated data with outputs filled in by converter
    """
    from .minima_llm import MinimaLlmConfig  # Import here to avoid circular

    if not data:
        return data

    return asyncio.run(
        run_dspy_batch(
            signature,
            data,
            converter,
            backend=OpenAIMinimaLlm(llm_config),
        )
    )


def print_dspy_prompt(sig:dspy.Signature, inputs:Dict[str,Any]):
    predict = dspy.Predict(sig)

    adapter = dspy.settings.adapter

    messages = adapter.format(
        signature=predict.signature,
        demos=[],               # no few-shot examples
        inputs=inputs
    )

    print(messages)
