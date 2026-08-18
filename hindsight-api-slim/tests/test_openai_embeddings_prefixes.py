"""
Tests for query_prefix and passage_prefix configuration on OpenAIEmbeddings.
"""

import os
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def setup_test_env():
    """Save/restore env vars touched by these tests."""
    from hindsight_api.config import clear_config_cache

    env_vars_to_save = [
        "HINDSIGHT_API_EMBEDDINGS_PROVIDER",
        "HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY",
        "HINDSIGHT_API_EMBEDDINGS_OPENAI_QUERY_PREFIX",
        "HINDSIGHT_API_EMBEDDINGS_OPENAI_PASSAGE_PREFIX",
        "HINDSIGHT_API_LLM_PROVIDER",
    ]

    original_values = {key: os.environ.get(key) for key in env_vars_to_save}
    clear_config_cache()

    yield

    for key, original_value in original_values.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value

    clear_config_cache()


def test_openai_embeddings_prefixes_default_to_empty():
    """Test that query_prefix and passage_prefix default to empty strings."""
    from hindsight_api.config import HindsightConfig
    from hindsight_api.engine.embeddings import OpenAIEmbeddings

    config = HindsightConfig.from_env()
    assert config.embeddings_openai_query_prefix == ""
    assert config.embeddings_openai_passage_prefix == ""

    client = OpenAIEmbeddings(api_key="test-key")
    assert client.query_prefix == ""
    assert client.passage_prefix == ""


def test_openai_embeddings_prefixes_from_env():
    """Test loading query_prefix and passage_prefix from env vars via create_embeddings_from_env."""
    from hindsight_api.engine.embeddings import OpenAIEmbeddings, create_embeddings_from_env

    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    os.environ["HINDSIGHT_API_EMBEDDINGS_PROVIDER"] = "openai"
    os.environ["HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY"] = "sk-test"
    os.environ["HINDSIGHT_API_EMBEDDINGS_OPENAI_QUERY_PREFIX"] = "query: "
    os.environ["HINDSIGHT_API_EMBEDDINGS_OPENAI_PASSAGE_PREFIX"] = "passage: "

    embeddings = create_embeddings_from_env()
    assert isinstance(embeddings, OpenAIEmbeddings)
    assert embeddings.query_prefix == "query: "
    assert embeddings.passage_prefix == "passage: "


def test_openai_embeddings_encode_query_and_documents_apply_prefixes():
    """Test encode_query and encode_documents apply prefixes before calling encode."""
    from hindsight_api.engine.embeddings import OpenAIEmbeddings

    client = OpenAIEmbeddings(
        api_key="test-key",
        query_prefix="search_query: ",
        passage_prefix="search_document: ",
    )
    client.encode = MagicMock(return_value=[[0.1, 0.2]])

    res_q = client.encode_query(["hello world"])
    client.encode.assert_called_with(["search_query: hello world"])
    assert res_q == [[0.1, 0.2]]

    res_doc = client.encode_documents(["sample document"])
    client.encode.assert_called_with(["search_document: sample document"])
    assert res_doc == [[0.1, 0.2]]
