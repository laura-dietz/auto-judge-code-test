"""
Integration tests for MinimaLLM core functionality.

These tests verify the code doesn't crash when used normally and all responses
are returned correctly. They make real LLM calls using environment-based config.

Prerequisites:
- OPENAI_BASE_URL must be set
- OPENAI_MODEL must be set
- OPENAI_API_KEY (or OPENAI_TOKEN) must be set
"""

import pytest
from trec_auto_judge.llm import (
    BatchConfig,
    MinimaLlmConfig,
    MinimaLlmRequest,
    MinimaLlmResponse,
    OpenAIMinimaLlm,
)


@pytest.mark.asyncio
async def test_single_request():
    """Test single LLM request doesn't crash"""
    llm = OpenAIMinimaLlm.from_env()

    req = MinimaLlmRequest(
        request_id="test-1",
        messages=[{"role": "user", "content": "Say hello"}],
    )

    response = await llm.generate(req)

    assert isinstance(response, MinimaLlmResponse)
    assert response.request_id == "test-1"
    assert len(response.text) > 0

    await llm.aclose()


@pytest.mark.asyncio
async def test_batch_execution():
    """Test parallel batch execution returns all responses"""
    llm = OpenAIMinimaLlm.from_env()

    requests = [
        MinimaLlmRequest(request_id="req-1", messages=[{"role": "user", "content": "Say hello"}]),
        MinimaLlmRequest(request_id="req-2", messages=[{"role": "user", "content": "Count to 3"}]),
        MinimaLlmRequest(request_id="req-3", messages=[{"role": "user", "content": "Name a color"}]),
    ]

    results = await llm.run_batched(requests)

    # All responses returned
    assert len(results) == 3

    # All are successes (not failures)
    for result in results:
        assert isinstance(result, MinimaLlmResponse), f"Got failure: {result}"

    # Request IDs preserved
    returned_ids = {r.request_id for r in results}
    expected_ids = {"req-1", "req-2", "req-3"}
    assert returned_ids == expected_ids

    await llm.aclose()


def test_config_loading():
    """Test configuration loads from environment"""
    llm_config = MinimaLlmConfig.from_env()
    batch_config = BatchConfig.from_env()

    # Basic sanity checks
    assert llm_config.base_url
    assert llm_config.model
    assert llm_config.batch is not None
    assert batch_config.num_workers > 0


@pytest.mark.asyncio
async def test_prompt_cache(tmp_path):
    """Test SQLite prompt cache works correctly"""
    import os

    cache_dir = str(tmp_path / "cache")

    # Create config with caching enabled
    base_config = MinimaLlmConfig.from_env()
    config = MinimaLlmConfig(
        base_url=base_config.base_url,
        model=base_config.model,
        api_key=base_config.api_key,
        cache_dir=cache_dir,
    )

    llm = OpenAIMinimaLlm(config)

    req = MinimaLlmRequest(
        request_id="cache-test-1",
        messages=[{"role": "user", "content": "Say the word 'cached'"}],
    )

    # First request - makes actual LLM call
    response1 = await llm.generate(req)
    assert isinstance(response1, MinimaLlmResponse)

    # Verify cache file was created
    db_path = os.path.join(cache_dir, "minima_llm.db")
    assert os.path.exists(db_path), "Cache DB should be created"

    # Second request with same content - should return from cache
    req2 = MinimaLlmRequest(
        request_id="cache-test-2",  # Different request_id, same content
        messages=[{"role": "user", "content": "Say the word 'cached'"}],
    )
    response2 = await llm.generate(req2)

    # Same text response (from cache), but request_id should be the new one
    assert response2.text == response1.text, "Cached response should match"
    assert response2.request_id == "cache-test-2", "Request ID should be from new request"

    await llm.aclose()
