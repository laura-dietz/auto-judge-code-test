
from dataclasses import dataclass
import os
from typing import Optional


@dataclass(frozen=True)
class MinimaLlmConfig:
    """
    Construct a MinimaLlmConfig from environment variables.

    This method is the single entry point for configuring MinimaLlm.
    All defaults are defined here and can be overridden via environment
    variables, making batch runs reproducible and easy to tune without
    code changes.

    Required environment variables
    ------------------------------
    - OPENAI_BASE_URL
        Base URL of an OpenAI-compatible endpoint (with or without `/v1`).

    - OPENAI_MODEL
        Model identifier understood by the endpoint.

    Optional authentication
    -----------------------
    - OPENAI_API_KEY or OPENAI_TOKEN
        Bearer token used for authentication.
        May be omitted for local or unsecured endpoints.

    Batch execution and monitoring
    ------------------------------
    - BATCH_NUM_WORKERS
        Number of concurrent batch workers (default: 64).

    - BATCH_MAX_FAILURES
        Abort the batch after this many failures.
        Set to an integer value, or to `none` / `null` to disable early abort
        (default: 25).

    - BATCH_HEARTBEAT_S
        Interval in seconds at which batch progress is printed
        (default: 10.0).

    - BATCH_STALL_S
        Emit a stall warning if no completions occur for this many seconds
        (default: 300.0).

    - BATCH_PRINT_FIRST_FAILURES
        Number of initial failures that are printed verbosely
        (default: 5).

    - BATCH_KEEP_FAILURE_SUMMARIES
        Number of recent failure summaries kept for abort diagnostics
        (default: 20).

    Transport-level protection
    --------------------------
    - MAX_OUTSTANDING
        Maximum number of in-flight HTTP requests (default: 32).

    - RPM
        Maximum requests per minute; set to 0 to disable pacing
        (default: 600).

    - TIMEOUT_S
        Per-request timeout in seconds (default: 60.0).

    Retry, backoff, and cooldown
    ----------------------------
    - MAX_ATTEMPTS
        Maximum number of attempts per request (default: 6).

    - BASE_BACKOFF_S
        Base delay for exponential backoff (default: 0.5).

    - MAX_BACKOFF_S
        Maximum backoff delay in seconds (default: 20.0).

    - JITTER
        Proportional jitter applied to backoff delays (default: 0.2).

    - COOLDOWN_FLOOR_S
        Minimum cooldown applied after overload signals (default: 0.0).

    - COOLDOWN_CAP_S
        Maximum cooldown delay in seconds (default: 30.0).

    - COOLDOWN_HALFLIFE_S
        Half-life (in seconds) for cooldown decay after overload
        (default: 20.0).

    HTTP behavior
    -------------
    - COMPRESS_GZIP
        If non-zero, request bodies are gzip-compressed.
        Disabled by default for compatibility with many endpoints.
    """

    # Endpoint
    base_url: str
    model: str
    api_key: Optional[str] = None  # optional for local endpoints

    # Batch execution
    num_workers: int = 64

    # Transport / backpressure
    max_outstanding: int = 32
    rpm: int = 600  # 0 disables pacing
    timeout_s: float = 60.0

    # Retry/backoff
    max_attempts: int = 6
    base_backoff_s: float = 0.5
    max_backoff_s: float = 20.0
    jitter: float = 0.2

    # Cooldown after overload (429/503/504)
    cooldown_floor_s: float = 0.0
    cooldown_cap_s: float = 30.0
    cooldown_halflife_s: float = 20.0

    # HTTP
    compress_gzip: bool = False


    # Batch execution (client-side)
    num_workers: int = 64

    # Batch failure policy
    max_failures: Optional[int] = 25
    print_first_failures: int = 5
    keep_failure_summaries: int = 20

    # Heartbeat / stall detection
    heartbeat_s: float = 10.0
    stall_s: float = 300.0
    
    # ----------------------------
    # Config parsing (private)
    # ----------------------------

    @classmethod
    def _env_opt_int(cls, name: str, default: Optional[int]) -> Optional[int]:
        v = cls._env_str(name)
        if v is None:
            return default
        if v.strip().lower() in ("none", "null"):
            return None
        return int(v)

    @staticmethod
    def _first_non_none(*vals: Optional[str]) -> Optional[str]:
        for v in vals:
            if v is not None:
                return v
        return None

    @staticmethod
    def _env_str(name: str) -> Optional[str]:
        v = os.getenv(name)
        return None if v in (None, "") else v

    @classmethod
    def _env_int(cls, name: str, default: int) -> int:
        v = cls._env_str(name)
        return default if v is None else int(v)

    @classmethod
    def _env_float(cls, name: str, default: float) -> float:
        v = cls._env_str(name)
        return default if v is None else float(v)

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        return base_url.rstrip("/")

    # ----------------------------
    # Public constructors
    # ----------------------------

    @classmethod
    def from_env(cls) -> "MinimaLlmConfig":
        """
        Construct a MinimaLlmConfig from environment variables.

        Required:
          - OPENAI_BASE_URL
          - OPENAI_MODEL
        """
        base_url = cls._env_str("OPENAI_BASE_URL")
        model = cls._env_str("OPENAI_MODEL")
        api_key = cls._first_non_none(
            cls._env_str("OPENAI_API_KEY"),
            cls._env_str("OPENAI_TOKEN"),
        )

        missing = []
        if base_url is None:
            missing.append("OPENAI_BASE_URL")
        if model is None:
            missing.append("OPENAI_MODEL")
        if missing:
            raise RuntimeError(f"Missing required environment variable(s): {', '.join(missing)}")

        return cls(
            base_url=cls._normalize_base_url(base_url),
            model=model,
            api_key=api_key,
            # 
            num_workers=cls._env_int("BATCH_NUM_WORKERS", 64),
            max_outstanding=cls._env_int("MAX_OUTSTANDING", 32),
            max_failures=cls._env_opt_int("BATCH_MAX_FAILURES", 25),
            heartbeat_s=cls._env_float("BATCH_HEARTBEAT_S", 10.0),
            stall_s=cls._env_float("BATCH_STALL_S", 300.0),
            print_first_failures=cls._env_int("BATCH_PRINT_FIRST_FAILURES", 5),
            keep_failure_summaries=cls._env_int("BATCH_KEEP_FAILURE_SUMMARIES", 20),
            #
            rpm=cls._env_int("RPM", 600),
            timeout_s=cls._env_float("TIMEOUT_S", 60.0),
            max_attempts=cls._env_int("MAX_ATTEMPTS", 6),
            base_backoff_s=cls._env_float("BASE_BACKOFF_S", 0.5),
            max_backoff_s=cls._env_float("MAX_BACKOFF_S", 20.0),
            jitter=cls._env_float("JITTER", 0.2),
            cooldown_floor_s=cls._env_float("COOLDOWN_FLOOR_S", 0.0),
            cooldown_cap_s=cls._env_float("COOLDOWN_CAP_S", 30.0),
            cooldown_halflife_s=cls._env_float("COOLDOWN_HALFLIFE_S", 20.0),
            compress_gzip=(cls._env_int("COMPRESS_GZIP", 0) != 0),

        )
        
    def describe(self) -> str:
        """
        Return a human-readable description of the active MinimaLlm configuration.

        This is intended for logging at startup of long-running batch jobs,
        so that execution parameters are recorded alongside results.
        """
        lines = []

        def add(section: str) -> None:
            lines.append(section)

        def kv(k: str, v) -> None:
            lines.append(f"  {k}: {v}")

        add("MinimaLlmConfig")
        add("Endpoint")
        kv("base_url", self.base_url)
        kv("model", self.model)
        kv("api_key", "<set>" if self.api_key is not None else "<none>")

        add("Batch execution")
        kv("num_workers", self.num_workers)
        kv("max_failures", self.max_failures)
        kv("heartbeat_s", self.heartbeat_s)
        kv("stall_s", self.stall_s)
        kv("print_first_failures", self.print_first_failures)
        kv("keep_failure_summaries", self.keep_failure_summaries)

        add("Transport / pacing")
        kv("max_outstanding", self.max_outstanding)
        kv("rpm", self.rpm)
        kv("timeout_s", self.timeout_s)

        add("Retry / backoff")
        kv("max_attempts", self.max_attempts)
        kv("base_backoff_s", self.base_backoff_s)
        kv("max_backoff_s", self.max_backoff_s)
        kv("jitter", self.jitter)

        add("Cooldown")
        kv("cooldown_floor_s", self.cooldown_floor_s)
        kv("cooldown_cap_s", self.cooldown_cap_s)
        kv("cooldown_halflife_s", self.cooldown_halflife_s)

        add("HTTP")
        kv("compress_gzip", self.compress_gzip)

        return "\n".join(lines)
