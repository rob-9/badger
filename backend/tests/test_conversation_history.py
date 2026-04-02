"""Tests for conversation history formatting and API validation."""

import pytest
from pydantic import ValidationError

from badger.api.server import ConversationTurn, QueryBookRequest
from badger.core.agent import _format_conversation_history


# ── _format_conversation_history ──────────────────────────────────────


class TestFormatConversationHistory:
    """Edge cases for the history formatter injected into the system prompt."""

    def test_empty_history(self):
        assert _format_conversation_history([]) == ""

    def test_none_values_in_turns(self):
        history = [
            {"role": "user", "content": "Who is X?", "selected_text": None, "reader_position": None},
            {"role": "assistant", "content": "X is a character."},
        ]
        result = _format_conversation_history(history)
        assert "Q: Who is X?" in result
        assert "A: X is a character." in result
        # No position or selected text header
        assert "Reader at" not in result
        assert "Selected:" not in result

    def test_basic_qa_pair(self):
        history = [
            {"role": "user", "content": "Who is Eidhin?", "selected_text": "He looked baffled", "reader_position": 0.3},
            {"role": "assistant", "content": "Eidhin is a monk who works as a translator."},
        ]
        result = _format_conversation_history(history)
        assert "[CONVERSATION HISTORY]" in result
        assert "Reader at 30%" in result
        assert 'Selected: "He looked baffled"' in result
        assert "Q: Who is Eidhin?" in result
        assert "A: Eidhin is a monk" in result

    def test_strips_source_citations(self):
        history = [
            {"role": "user", "content": "What happened?"},
            {"role": "assistant", "content": "He left the room [Source 1] and went outside [Source 2, Source 3]."},
        ]
        result = _format_conversation_history(history)
        assert "[Source" not in result
        assert "He left the room" in result
        assert "went outside" in result

    def test_strips_compact_citations(self):
        history = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "Answer [1] with [2,3] citations."},
        ]
        result = _format_conversation_history(history)
        assert "[1]" not in result
        assert "[2,3]" not in result
        assert "Answer" in result

    def test_truncates_long_answer(self):
        long_answer = "A" * 500
        history = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": long_answer},
        ]
        result = _format_conversation_history(history, max_answer_len=200)
        # Should be truncated with ellipsis
        assert "..." in result
        assert "A" * 201 not in result

    def test_truncates_long_selected_text(self):
        long_sel = "B" * 200
        history = [
            {"role": "user", "content": "Q", "selected_text": long_sel},
            {"role": "assistant", "content": "A"},
        ]
        result = _format_conversation_history(history, max_selected_len=80)
        assert "B" * 81 not in result
        assert "..." in result

    def test_max_turns_limits_output(self):
        history = []
        for i in range(10):
            history.append({"role": "user", "content": f"Question {i}"})
            history.append({"role": "assistant", "content": f"Answer {i}"})
        result = _format_conversation_history(history, max_turns=3)
        # Should only contain the last 3 exchanges (7, 8, 9)
        assert "Question 7" in result
        assert "Question 8" in result
        assert "Question 9" in result
        assert "Question 6" not in result
        assert "Question 0" not in result

    def test_all_user_messages_no_assistants(self):
        history = [
            {"role": "user", "content": "First question"},
            {"role": "user", "content": "Second question"},
        ]
        result = _format_conversation_history(history)
        # Both should be included as orphan user messages
        assert "First question" in result
        assert "Second question" in result

    def test_orphan_assistant_at_start_dropped(self):
        history = [
            {"role": "assistant", "content": "I am an orphan answer"},
            {"role": "user", "content": "Real question"},
            {"role": "assistant", "content": "Real answer"},
        ]
        result = _format_conversation_history(history)
        assert "orphan answer" not in result
        assert "Real question" in result
        assert "Real answer" in result

    def test_trailing_user_without_assistant(self):
        history = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Follow-up with no response yet"},
        ]
        result = _format_conversation_history(history)
        assert "Follow-up with no response yet" in result

    def test_pydantic_models_accepted(self):
        """The formatter should handle Pydantic ConversationTurn objects."""
        turns = [
            ConversationTurn(role="user", content="Who is X?", selected_text="some text", reader_position=0.5),
            ConversationTurn(role="assistant", content="X is a character."),
        ]
        result = _format_conversation_history(turns)
        assert "Q: Who is X?" in result
        assert "A: X is a character." in result
        assert "Reader at 50%" in result

    def test_reader_position_zero_shows_zero(self):
        history = [
            {"role": "user", "content": "Q", "reader_position": 0.0},
            {"role": "assistant", "content": "A"},
        ]
        result = _format_conversation_history(history)
        assert "Reader at 0%" in result

    def test_single_user_message_only(self):
        history = [{"role": "user", "content": "Just a question"}]
        result = _format_conversation_history(history)
        assert "Just a question" in result
        assert "[Turn 1]" in result

    def test_mixed_pydantic_and_dict(self):
        turns = [
            ConversationTurn(role="user", content="Dict-style Q"),
            {"role": "assistant", "content": "Dict answer"},
        ]
        result = _format_conversation_history(turns)
        assert "Dict-style Q" in result
        assert "Dict answer" in result


# ── ConversationTurn Pydantic validation ──────────────────────────────


class TestConversationTurnValidation:
    """Validators on the ConversationTurn model."""

    def test_valid_turn(self):
        turn = ConversationTurn(role="user", content="Hello")
        assert turn.role == "user"
        assert turn.content == "Hello"

    def test_empty_content_rejected(self):
        with pytest.raises(ValidationError, match="Content must not be empty"):
            ConversationTurn(role="user", content="")

    def test_whitespace_content_rejected(self):
        with pytest.raises(ValidationError, match="Content must not be empty"):
            ConversationTurn(role="user", content="   ")

    def test_content_over_5000_rejected(self):
        with pytest.raises(ValidationError, match="Content exceeds maximum length"):
            ConversationTurn(role="user", content="x" * 5001)

    def test_content_at_5000_accepted(self):
        turn = ConversationTurn(role="user", content="x" * 5000)
        assert len(turn.content) == 5000

    def test_invalid_role_rejected(self):
        with pytest.raises(ValidationError):
            ConversationTurn(role="system", content="Hello")

    def test_selected_text_too_long_rejected(self):
        from badger import config
        with pytest.raises(ValidationError, match="Selected text exceeds"):
            ConversationTurn(role="user", content="Q", selected_text="x" * (config.MAX_SELECTED_TEXT_LENGTH + 1))

    def test_reader_position_out_of_range(self):
        with pytest.raises(ValidationError, match="Reader position must be between"):
            ConversationTurn(role="user", content="Q", reader_position=1.5)

        with pytest.raises(ValidationError, match="Reader position must be between"):
            ConversationTurn(role="user", content="Q", reader_position=-0.1)

    def test_reader_position_at_boundaries(self):
        t0 = ConversationTurn(role="user", content="Q", reader_position=0.0)
        assert t0.reader_position == 0.0
        t1 = ConversationTurn(role="user", content="Q", reader_position=1.0)
        assert t1.reader_position == 1.0

    def test_optional_fields_default_none(self):
        turn = ConversationTurn(role="assistant", content="Answer")
        assert turn.selected_text is None
        assert turn.reader_position is None


# ── QueryBookRequest conversation_history validation ──────────────────


class TestQueryBookRequestHistoryValidation:
    """Validators on the conversation_history field of QueryBookRequest."""

    def test_history_optional_defaults_none(self):
        req = QueryBookRequest(question="What?")
        assert req.conversation_history is None

    def test_valid_history(self):
        req = QueryBookRequest(
            question="Follow-up?",
            conversation_history=[
                ConversationTurn(role="user", content="First Q"),
                ConversationTurn(role="assistant", content="First A"),
            ],
        )
        assert len(req.conversation_history) == 2

    def test_history_over_10_turns_rejected(self):
        turns = [
            ConversationTurn(role="user", content=f"Q{i}")
            for i in range(11)
        ]
        with pytest.raises(ValidationError, match="Conversation history exceeds maximum of 10"):
            QueryBookRequest(question="Too long?", conversation_history=turns)

    def test_history_at_10_turns_accepted(self):
        turns = [
            ConversationTurn(role="user", content=f"Q{i}")
            for i in range(10)
        ]
        req = QueryBookRequest(question="Just right?", conversation_history=turns)
        assert len(req.conversation_history) == 10

    def test_empty_history_accepted(self):
        req = QueryBookRequest(question="Q", conversation_history=[])
        assert req.conversation_history == []
