"""Tests for text chunker."""

import pytest

from src.utils.chunker import Chunker


def test_chunker_basic():
    """Test basic chunking."""
    chunker = Chunker(chunk_size=100, chunk_overlap=20)

    text = "This is a test sentence. And another one. Here is more text."
    chunks = chunker.split(text)

    assert len(chunks) >= 1
    assert all(c.text for c in chunks)
    assert all(c.token_count > 0 for c in chunks)


def test_chunker_empty():
    """Test empty text."""
    chunker = Chunker()

    assert chunker.split("") == []
    assert chunker.split("   ") == []


def test_chunker_single_sentence():
    """Test single sentence."""
    chunker = Chunker(chunk_size=1000)

    text = "This is a single sentence."
    chunks = chunker.split(text)

    assert len(chunks) == 1
    assert chunks[0].text == text


def test_chunker_long_sentence():
    """Test sentence exceeding chunk size."""
    chunker = Chunker(chunk_size=10, chunk_overlap=2)

    text = "This is a very long sentence that exceeds the chunk size limit."
    chunks = chunker.split(text)

    assert len(chunks) > 1


def test_count_tokens():
    """Test token counting."""
    chunker = Chunker()

    assert chunker.count_tokens("Hello world") > 0
    assert chunker.count_tokens("") == 0
