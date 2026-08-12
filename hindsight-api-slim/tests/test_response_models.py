"""Tests for MemoryFact response-model validation."""

from hindsight_api.engine.response_models import MemoryFact

_BASE = {"id": "u1", "text": "hi", "fact_type": "observation"}


class TestMemoryFactMetadataCoercion:
    """metadata values may arrive as non-strings from JSONB; they must coerce to str.

    Regression for #2622: an integer ``original_id`` stored in ``metadata`` raised
    ``ValidationError`` (``string_type``) and blocked consolidation, because
    ``metadata`` is typed ``dict[str, str]`` and the raw dict passed straight
    through to Pydantic.
    """

    def test_integer_value_is_coerced_to_string(self):
        fact = MemoryFact(**_BASE, metadata={"original_id": 12345})
        assert fact.metadata == {"original_id": "12345"}

    def test_jsonb_string_with_integer_value_is_coerced(self):
        # asyncpg may hand back JSONB as a raw string; ints inside must still coerce.
        fact = MemoryFact(**_BASE, metadata='{"original_id": 12345, "n": 3}')
        assert fact.metadata == {"original_id": "12345", "n": "3"}

    def test_string_values_pass_through_unchanged(self):
        fact = MemoryFact(**_BASE, metadata={"source": "slack", "channel": "eng"})
        assert fact.metadata == {"source": "slack", "channel": "eng"}

    def test_none_metadata_stays_none(self):
        assert MemoryFact(**_BASE, metadata=None).metadata is None

    def test_null_valued_key_is_dropped(self):
        """Regression for #3209: metadata with null values must not raise ValidationError.

        asyncpg/JSONB can return {"key": null}; previously str(None) -> "None" was wrong.
        The fix drops null-valued keys so recall of pre-existing rows stays readable.
        """
        fact = MemoryFact(**_BASE, metadata={"present": "yes", "absent": None})
        assert fact.metadata == {"present": "yes"}

    def test_null_valued_key_dropped_via_json_string(self):
        """Same as above but arriving as a raw JSON string (asyncpg JSONB-as-str path)."""
        fact = MemoryFact(**_BASE, metadata='{"present": "yes", "absent": null}')
        assert fact.metadata == {"present": "yes"}

    def test_all_null_values_gives_empty_dict(self):
        fact = MemoryFact(**_BASE, metadata={"a": None, "b": None})
        assert fact.metadata == {}
