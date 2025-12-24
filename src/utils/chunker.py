"""Text chunking utilities."""

import re
from dataclasses import dataclass

import tiktoken

from src.config import settings


@dataclass
class ChunkResult:
    """Result of chunking operation."""

    text: str
    start_char: int
    end_char: int
    token_count: int


class Chunker:
    """
    Smart text chunker with overlap.

    Respects sentence boundaries when possible.
    Uses tiktoken for accurate token counting.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self._encoder = tiktoken.get_encoding("cl100k_base")

        # Sentence splitters (multilingual)
        self._sentence_pattern = re.compile(
            r"(?<=[.!?。！？])\s+|(?<=[.!?。！？])(?=[A-ZА-ЯЁ])"
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self._encoder.encode(text))

    def split(self, text: str) -> list[ChunkResult]:
        """
        Split text into chunks.

        Strategy:
        1. Split by sentences
        2. Accumulate sentences until chunk_size reached
        3. Add overlap from previous chunk
        """
        if not text or not text.strip():
            return []

        # Split into sentences
        sentences = self._sentence_pattern.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return [
                ChunkResult(
                    text=text.strip(),
                    start_char=0,
                    end_char=len(text),
                    token_count=self.count_tokens(text),
                )
            ]

        chunks: list[ChunkResult] = []
        current_sentences: list[str] = []
        current_tokens = 0
        current_start = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            # If single sentence exceeds chunk_size, split it
            if sentence_tokens > self.chunk_size:
                # Flush current chunk if exists
                if current_sentences:
                    chunk_text = " ".join(current_sentences)
                    chunks.append(
                        ChunkResult(
                            text=chunk_text,
                            start_char=current_start,
                            end_char=current_start + len(chunk_text),
                            token_count=current_tokens,
                        )
                    )
                    current_start += len(chunk_text) + 1

                # Split long sentence by tokens
                words = sentence.split()
                word_chunks = self._split_long_sentence(words)
                for wc in word_chunks:
                    chunks.append(
                        ChunkResult(
                            text=wc,
                            start_char=current_start,
                            end_char=current_start + len(wc),
                            token_count=self.count_tokens(wc),
                        )
                    )
                    current_start += len(wc) + 1

                current_sentences = []
                current_tokens = 0
                continue

            # Check if adding sentence exceeds chunk_size
            if current_tokens + sentence_tokens > self.chunk_size:
                # Flush current chunk
                if current_sentences:
                    chunk_text = " ".join(current_sentences)
                    chunks.append(
                        ChunkResult(
                            text=chunk_text,
                            start_char=current_start,
                            end_char=current_start + len(chunk_text),
                            token_count=current_tokens,
                        )
                    )

                    # Calculate overlap
                    overlap_sentences = self._get_overlap_sentences(
                        current_sentences, self.chunk_overlap
                    )
                    current_sentences = overlap_sentences
                    current_tokens = sum(
                        self.count_tokens(s) for s in current_sentences
                    )
                    current_start += len(chunk_text) - sum(
                        len(s) + 1 for s in overlap_sentences
                    )

            current_sentences.append(sentence)
            current_tokens += sentence_tokens

        # Flush remaining
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(
                ChunkResult(
                    text=chunk_text,
                    start_char=current_start,
                    end_char=current_start + len(chunk_text),
                    token_count=current_tokens,
                )
            )

        return chunks

    def _split_long_sentence(self, words: list[str]) -> list[str]:
        """Split a long sentence that exceeds chunk_size."""
        chunks = []
        current_words: list[str] = []
        current_tokens = 0

        for word in words:
            word_tokens = self.count_tokens(word)
            if current_tokens + word_tokens > self.chunk_size:
                if current_words:
                    chunks.append(" ".join(current_words))
                current_words = [word]
                current_tokens = word_tokens
            else:
                current_words.append(word)
                current_tokens += word_tokens

        if current_words:
            chunks.append(" ".join(current_words))

        return chunks

    def _get_overlap_sentences(
        self, sentences: list[str], overlap_tokens: int
    ) -> list[str]:
        """Get sentences from the end that fit in overlap_tokens."""
        overlap: list[str] = []
        total_tokens = 0

        for sentence in reversed(sentences):
            sentence_tokens = self.count_tokens(sentence)
            if total_tokens + sentence_tokens <= overlap_tokens:
                overlap.insert(0, sentence)
                total_tokens += sentence_tokens
            else:
                break

        return overlap
