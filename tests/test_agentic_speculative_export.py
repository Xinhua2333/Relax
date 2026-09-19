# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from typing import Any

import pytest

from relax.agentic.session.state import MsgNode, SessionForest
from relax.utils.speculative import SpeculativeCounts


class Tokenizer:
    def decode(self, tokens: list[int], **kwargs: Any) -> str:
        return "".join(chr(token) for token in tokens)


def forest_with_prompt(session_id: str = "session") -> tuple[SessionForest, MsgNode]:
    forest = SessionForest.create_empty(session_id=session_id)
    prompt = forest.append_obs(
        parent_state_hash=forest.root_state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=[{"role": "user", "content": "prompt"}],
        train_token_delta=[112],
        rollout_token_delta=[112],
    )
    return forest, prompt


def commit_response(
    forest: SessionForest,
    parent: MsgNode,
    request_id: str,
    text: str = "a",
    counts: SpeculativeCounts | None = None,
) -> MsgNode:
    return forest.append_resp(
        parent_state_hash=parent.state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=[{"role": "assistant", "content": text}],
        token_delta=[ord(char) for char in text],
        logprob_delta=[-0.1] * len(text),
        spec_counts=counts,
        export_metadata_patch={"request_id": request_id},
    )


def test_speculative_identity_separates_equal_content_and_is_idempotent() -> None:
    forest, prompt = forest_with_prompt()
    first = commit_response(forest, prompt, "one", counts=SpeculativeCounts(1, 2, 1, 2))
    second = commit_response(forest, prompt, "two", counts=SpeculativeCounts(9, 10, 2, 11))
    assert first is second  # Preserve existing content-addressed matching.
    commit_response(forest, prompt, "two", counts=SpeculativeCounts(9, 10, 2, 11))
    assert forest.generation_ids_by_state[first.state_hash] == ["one", "two"]
    assert len(forest.committed_generations) == 2
    assert forest.committed_generations["one"].counts.accepted == 1
    assert forest.committed_generations["two"].counts.accepted == 9


def test_speculative_identity_conflict_does_not_register_a_state() -> None:
    forest, prompt = forest_with_prompt()
    commit_response(forest, prompt, "one", counts=SpeculativeCounts(1, 2))
    leaves = forest.export_leaf_hashes()
    with pytest.raises(ValueError, match="Conflicting committed generation"):
        commit_response(forest, prompt, "one", text="different", counts=SpeculativeCounts(1, 2))
    with pytest.raises(ValueError, match="Conflicting committed generation"):
        commit_response(forest, prompt, "one", counts=SpeculativeCounts(2, 2))
    assert forest.export_leaf_hashes() == leaves


def test_speculative_identity_sessions_and_uncommitted_requests() -> None:
    first, prompt = forest_with_prompt("first")
    second, other_prompt = forest_with_prompt("second")
    assert first.committed_generations == {}
    commit_response(first, prompt, "same-id")
    commit_response(second, other_prompt, "same-id")
    assert first.committed_generations["same-id"].session_id == "first"
    assert second.committed_generations["same-id"].session_id == "second"
