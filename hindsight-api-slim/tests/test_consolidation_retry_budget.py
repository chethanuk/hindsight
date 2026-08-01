"""Tests for consolidation retry budget configurability (issue #1042)."""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api.engine.consolidation.consolidator import _consolidate_batch_with_llm


@pytest.fixture
def mock_llm_config():
    llm = AsyncMock()
    response = MagicMock()
    response.creates = []
    response.updates = []
    response.deletes = []
    llm.call.return_value = response
    return llm


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.observations_mission = None
    config.consolidation_max_attempts = 3
    config.consolidation_llm_max_retries = None
    config.consolidation_max_completion_tokens = None
    config.llm_strict_schema_consolidation = False
    return config


class TestConsolidationRetryBudget:
    @pytest.mark.asyncio
    async def test_config_is_required(self, mock_llm_config):
        """Passing config=None raises — it's a programmer error, not a runtime fallback."""
        with pytest.raises(ValueError, match="config is required"):
            await _consolidate_batch_with_llm(
                llm_config=mock_llm_config,
                memories=[{"id": "m1", "text": "test"}],
                union_observations=[],
                union_source_facts={},
                config=None,
            )

    @pytest.mark.asyncio
    async def test_configurable_max_attempts(self, mock_llm_config, mock_config):
        """consolidation_max_attempts controls the outer retry loop."""
        mock_config.consolidation_max_attempts = 5
        mock_llm_config.call.side_effect = RuntimeError("fail")
        result = await _consolidate_batch_with_llm(
            llm_config=mock_llm_config,
            memories=[{"id": "m1", "text": "test"}],
            union_observations=[],
            union_source_facts={},
            config=mock_config,
        )
        assert result.failed
        assert mock_llm_config.call.call_count == 5

    @pytest.mark.asyncio
    async def test_max_retries_threaded_to_call(self, mock_llm_config, mock_config):
        """consolidation_llm_max_retries is passed to llm_config.call()."""
        mock_config.consolidation_llm_max_retries = 3
        await _consolidate_batch_with_llm(
            llm_config=mock_llm_config,
            memories=[{"id": "m1", "text": "test"}],
            union_observations=[],
            union_source_facts={},
            config=mock_config,
        )
        assert mock_llm_config.call.call_args.kwargs.get("max_retries") == 3

    @pytest.mark.asyncio
    async def test_strict_schema_threaded_to_call(self, mock_llm_config, mock_config):
        """llm_strict_schema_consolidation is passed to llm_config.call()."""
        mock_config.llm_strict_schema_consolidation = True
        await _consolidate_batch_with_llm(
            llm_config=mock_llm_config,
            memories=[{"id": "m1", "text": "test"}],
            union_observations=[],
            union_source_facts={},
            config=mock_config,
        )
        assert mock_llm_config.call.call_args.kwargs.get("strict_schema") is True

    @pytest.mark.asyncio
    async def test_strict_schema_passed_as_explicit_false(self, mock_llm_config, mock_config):
        """A disabled per-operation flag is passed explicitly, not omitted.

        Omitting it would let the global HINDSIGHT_API_LLM_STRICT_SCHEMA flag win,
        which is exactly what the per-operation opt-out exists to prevent.
        """
        mock_config.llm_strict_schema_consolidation = False
        await _consolidate_batch_with_llm(
            llm_config=mock_llm_config,
            memories=[{"id": "m1", "text": "test"}],
            union_observations=[],
            union_source_facts={},
            config=mock_config,
        )
        assert mock_llm_config.call.call_args.kwargs.get("strict_schema") is False

    @pytest.mark.asyncio
    async def test_max_completion_tokens_threaded_to_call(self, mock_llm_config, mock_config):
        """consolidation_max_completion_tokens is passed to llm_config.call()."""
        mock_config.consolidation_max_completion_tokens = 8192
        await _consolidate_batch_with_llm(
            llm_config=mock_llm_config,
            memories=[{"id": "m1", "text": "test"}],
            union_observations=[],
            union_source_facts={},
            config=mock_config,
        )
        assert mock_llm_config.call.call_args.kwargs.get("max_completion_tokens") == 8192

    @pytest.mark.asyncio
    async def test_max_completion_tokens_not_passed_when_none(self, mock_llm_config, mock_config):
        """When consolidation_max_completion_tokens is None, max_completion_tokens is omitted (no regression)."""
        mock_config.consolidation_max_completion_tokens = None
        await _consolidate_batch_with_llm(
            llm_config=mock_llm_config,
            memories=[{"id": "m1", "text": "test"}],
            union_observations=[],
            union_source_facts={},
            config=mock_config,
        )
        assert "max_completion_tokens" not in mock_llm_config.call.call_args.kwargs

    @pytest.mark.asyncio
    async def test_max_retries_not_passed_when_none(self, mock_llm_config, mock_config):
        """When consolidation_llm_max_retries is None, max_retries is not passed."""
        mock_config.consolidation_llm_max_retries = None
        await _consolidate_batch_with_llm(
            llm_config=mock_llm_config,
            memories=[{"id": "m1", "text": "test"}],
            union_observations=[],
            union_source_facts={},
            config=mock_config,
        )
        assert "max_retries" not in mock_llm_config.call.call_args.kwargs

    @pytest.mark.asyncio
    async def test_partially_malformed_response_costs_no_attempts(self, mock_llm_config, mock_config, caplog):
        """#3003: a single malformed update is dropped, not retried.

        The provider parses the raw JSON through `response_format`, so the ValidationError
        used to surface here as a plain batch failure — valid actions discarded and every
        remaining attempt spent re-asking a model that answers the same way each time.
        """
        raw = {
            "creates": [{"text": "User is training for a marathon.", "source_fact_ids": ["m1"]}],
            "updates": [{"text": "User moved from Lisbon to Berlin in March."}],
            "deletes": [{"observation_id": "obs-stale"}],
        }

        async def parse_like_the_provider(**kwargs):
            return kwargs["response_format"].model_validate(raw)

        mock_llm_config.call.side_effect = parse_like_the_provider

        with caplog.at_level(logging.WARNING, logger="hindsight_api.engine.consolidation.consolidator"):
            result = await _consolidate_batch_with_llm(
                llm_config=mock_llm_config,
                memories=[{"id": "m1", "text": "test"}],
                union_observations=[],
                union_source_facts={},
                config=mock_config,
            )

        assert result.failed is False
        assert mock_llm_config.call.call_count == 1, "no retry attempts burned on a deterministic parse failure"
        assert [c.text for c in result.creates] == ["User is training for a marathon."]
        assert result.updates == []
        assert [d.observation_id for d in result.deletes] == ["obs-stale"]
        assert any("1 memories [m1]" in r.message for r in caplog.records if r.levelno == logging.WARNING), (
            "the drop warning must name the batch it came from"
        )

    @pytest.mark.asyncio
    async def test_reduced_budget_limits_total_calls(self, mock_llm_config, mock_config):
        """Setting both to low values caps total failure attempts."""
        mock_config.consolidation_max_attempts = 2
        mock_config.consolidation_llm_max_retries = 2
        mock_llm_config.call.side_effect = RuntimeError("upstream 503")
        result = await _consolidate_batch_with_llm(
            llm_config=mock_llm_config,
            memories=[{"id": "m1", "text": "test"}],
            union_observations=[],
            union_source_facts={},
            config=mock_config,
        )
        assert result.failed
        assert mock_llm_config.call.call_count == 2
        for call_args in mock_llm_config.call.call_args_list:
            assert call_args.kwargs.get("max_retries") == 2
