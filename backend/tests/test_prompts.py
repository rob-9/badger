"""Tests for boom.core.prompts — system prompts for the LangGraph pipeline."""

from boom.core.prompts import SYSTEM_PROMPTS, POSITION_INSTRUCTIONS


class TestPrompts:
    def test_all_question_types_have_prompts(self):
        expected = {"vocabulary", "context", "lookup", "analysis"}
        assert set(SYSTEM_PROMPTS.keys()) == expected

    def test_prompts_include_position_instructions(self):
        for q_type, prompt in SYSTEM_PROMPTS.items():
            assert "ALREADY READ" in prompt, f"{q_type} prompt missing ALREADY READ"
            assert "COMING UP" in prompt, f"{q_type} prompt missing COMING UP"

    def test_vocabulary_prompt_mentions_definition(self):
        assert "definition" in SYSTEM_PROMPTS["vocabulary"].lower()

    def test_context_prompt_mentions_passage(self):
        assert "passage" in SYSTEM_PROMPTS["context"].lower()

    def test_lookup_prompt_mentions_factual(self):
        assert "factual" in SYSTEM_PROMPTS["lookup"].lower()

    def test_analysis_prompt_mentions_themes(self):
        assert "themes" in SYSTEM_PROMPTS["analysis"].lower()

    def test_position_instructions_standalone(self):
        assert "ALREADY READ" in POSITION_INSTRUCTIONS
        assert "COMING UP" in POSITION_INSTRUCTIONS
        assert "spoil" in POSITION_INSTRUCTIONS.lower()
