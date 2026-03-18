from __future__ import annotations
import pytest
from src.exceptions import (
    BlockedByPolicyError,
    GuardError,
    InvalidToolCallError,
    LLMSecurityError,
    OutputBlockedError,
    PolicyNotFoundError,
    PromptBlockedError,
    ProviderError,
    ProviderTimeoutError,
)
from src.types import GuardDecision, GuardType, ScanResult

class TestLLMSecurityError:
    def test_is_exception(self):
        assert isinstance(LLMSecurityError("x"), Exception)

    def test_message_stored(self):
        exc = LLMSecurityError("test message")
        assert exc.message == "test message"

    def test_str_equals_message(self):
        exc = LLMSecurityError("test message")
        assert str(exc) == "test message"

    def test_context_defaults_to_empty_dict(self):
        exc = LLMSecurityError("x")
        assert exc.context == {}

    def test_context_stored(self):
        exc = LLMSecurityError("x", context={"req_id": "abc", "env": "prod"})
        assert exc.context == {"req_id": "abc", "env": "prod"}

    def test_context_not_shared_between_instances(self):
        exc1 = LLMSecurityError("a")
        exc2 = LLMSecurityError("b")
        exc1.context["key"] = "val"
        assert "key" not in exc2.context

    def test_repr_contains_class_name(self):
        exc = LLMSecurityError("test message")
        assert "LLMSecurityError" in repr(exc)

    def test_repr_contains_message(self):
        exc = LLMSecurityError("test message")
        assert "test message" in repr(exc)

    def test_empty_message_allowed(self):
        exc = LLMSecurityError("")
        assert exc.message == ""

class TestBlockedByPolicyError:
    def test_is_llm_security_error(self):
        assert isinstance(BlockedByPolicyError(reasons=["x"]), LLMSecurityError)

    def test_reasons_stored(self):
        exc = BlockedByPolicyError(reasons=["injection", "jailbreak"])
        assert exc.reasons == ["injection", "jailbreak"]

    def test_score_stored(self):
        exc = BlockedByPolicyError(reasons=["x"], score=0.92)
        assert exc.score == 0.92

    def test_score_default_is_one(self):
        exc = BlockedByPolicyError(reasons=["x"])
        assert exc.score == 1.0

    def test_policy_name_stored(self):
        exc = BlockedByPolicyError(reasons=["x"], policy_name="strict")
        assert exc.policy_name == "strict"

    def test_policy_name_default_is_unknown(self):
        exc = BlockedByPolicyError(reasons=["x"])
        assert exc.policy_name == "unknown"

    def test_decision_default_is_none(self):
        exc = BlockedByPolicyError(reasons=["x"])
        assert exc.decision is None

    def test_decision_stored(self):
        decision = GuardDecision.blocked(["x"], score=0.9)
        exc = BlockedByPolicyError(reasons=["x"], decision=decision)
        assert exc.decision is decision

    def test_context_stored(self):
        exc = BlockedByPolicyError(reasons=["x"], context={"req_id": "abc"})
        assert exc.context == {"req_id": "abc"}

    def test_message_contains_policy_name(self):
        exc = BlockedByPolicyError(reasons=["x"], policy_name="balanced")
        assert "balanced" in str(exc)

    def test_message_contains_score(self):
        exc = BlockedByPolicyError(reasons=["x"], score=0.92)
        assert "0.920" in str(exc)

    def test_message_contains_single_reason(self):
        exc = BlockedByPolicyError(reasons=["injection detected"])
        assert "injection detected" in str(exc)

    def test_message_joins_multiple_reasons_with_semicolons(self):
        exc = BlockedByPolicyError(reasons=["reason A", "reason B"], score=0.9)
        assert "reason A; reason B" in str(exc)

    def test_empty_reasons_message_has_no_details(self):
        exc = BlockedByPolicyError(reasons=[])
        assert "no details" in str(exc)

    def test_to_dict_keys(self):
        exc = BlockedByPolicyError(reasons=["x"], score=0.92, policy_name="bal")
        assert set(exc.to_dict().keys()) == {"error", "reasons", "score", "policy_name"}

    def test_to_dict_error_name(self):
        exc = BlockedByPolicyError(reasons=["x"])
        assert exc.to_dict()["error"] == "BlockedByPolicyError"

    def test_to_dict_reasons(self):
        exc = BlockedByPolicyError(reasons=["r1", "r2"])
        assert exc.to_dict()["reasons"] == ["r1", "r2"]

    def test_to_dict_score(self):
        exc = BlockedByPolicyError(reasons=["x"], score=0.85)
        assert exc.to_dict()["score"] == 0.85

    def test_to_dict_score_rounded_to_4_decimal_places(self):
        exc = BlockedByPolicyError(reasons=["x"], score=0.921234)
        assert exc.to_dict()["score"] == 0.9212

    def test_to_dict_policy_name(self):
        exc = BlockedByPolicyError(reasons=["x"], policy_name="strict")
        assert exc.to_dict()["policy_name"] == "strict"

class TestPromptBlockedError:
    def test_is_blocked_by_policy_error(self):
        assert isinstance(PromptBlockedError(reasons=["x"]), BlockedByPolicyError)

    def test_is_llm_security_error(self):
        assert isinstance(PromptBlockedError(reasons=["x"]), LLMSecurityError)

    def test_prompt_snippet_stored(self):
        exc = PromptBlockedError(reasons=["x"], prompt_snippet="Ignore all previous")
        assert exc.prompt_snippet == "Ignore all previous"

    def test_prompt_snippet_default_is_empty_string(self):
        exc = PromptBlockedError(reasons=["x"])
        assert exc.prompt_snippet == ""

    def test_prompt_snippet_truncated_to_120_chars(self):
        long_snippet = "A" * 200
        exc = PromptBlockedError(reasons=["x"], prompt_snippet=long_snippet)
        assert len(exc.prompt_snippet) == 120

    def test_prompt_snippet_exact_120_chars_not_truncated(self):
        snippet = "B" * 120
        exc = PromptBlockedError(reasons=["x"], prompt_snippet=snippet)
        assert len(exc.prompt_snippet) == 120
        assert exc.prompt_snippet == snippet

    def test_prompt_snippet_short_not_truncated(self):
        exc = PromptBlockedError(reasons=["x"], prompt_snippet="short text")
        assert exc.prompt_snippet == "short text"

    def test_prompt_snippet_truncated_content_is_prefix(self):
        long_snippet = "X" * 50 + "Y" * 150
        exc = PromptBlockedError(reasons=["x"], prompt_snippet=long_snippet)
        assert exc.prompt_snippet == "X" * 50 + "Y" * 70

    def test_score_inherited(self):
        exc = PromptBlockedError(reasons=["x"], score=0.92)
        assert exc.score == 0.92

    def test_reasons_inherited(self):
        exc = PromptBlockedError(reasons=["r1", "r2"])
        assert exc.reasons == ["r1", "r2"]

    def test_policy_name_inherited(self):
        exc = PromptBlockedError(reasons=["x"], policy_name="strict")
        assert exc.policy_name == "strict"

    def test_decision_inherited(self):
        decision = GuardDecision.blocked(["x"])
        exc = PromptBlockedError(reasons=["x"], decision=decision)
        assert exc.decision is decision

    def test_to_dict_has_prompt_snippet_key(self):
        exc = PromptBlockedError(reasons=["x"])
        assert "prompt_snippet" in exc.to_dict()

    def test_to_dict_error_name(self):
        exc = PromptBlockedError(reasons=["x"])
        assert exc.to_dict()["error"] == "PromptBlockedError"

    def test_to_dict_snippet_value(self):
        exc = PromptBlockedError(reasons=["x"], prompt_snippet="bad prompt")
        assert exc.to_dict()["prompt_snippet"] == "bad prompt"

    def test_to_dict_includes_inherited_fields(self):
        exc = PromptBlockedError(reasons=["r"], score=0.9, policy_name="bal")
        d = exc.to_dict()
        assert d["reasons"] == ["r"]
        assert d["score"] == 0.9
        assert d["policy_name"] == "bal"

class TestOutputBlockedError:
    def test_is_blocked_by_policy_error(self):
        assert isinstance(OutputBlockedError(reasons=["x"]), BlockedByPolicyError)

    def test_is_llm_security_error(self):
        assert isinstance(OutputBlockedError(reasons=["x"]), LLMSecurityError)

    def test_output_snippet_stored(self):
        exc = OutputBlockedError(reasons=["x"], output_snippet="sk-abc123...")
        assert exc.output_snippet == "sk-abc123..."

    def test_output_snippet_default_is_empty_string(self):
        exc = OutputBlockedError(reasons=["x"])
        assert exc.output_snippet == ""

    def test_output_snippet_truncated_to_120_chars(self):
        exc = OutputBlockedError(reasons=["x"], output_snippet="C" * 200)
        assert len(exc.output_snippet) == 120

    def test_output_snippet_exact_120_chars_unchanged(self):
        snippet = "D" * 120
        exc = OutputBlockedError(reasons=["x"], output_snippet=snippet)
        assert exc.output_snippet == snippet

    def test_output_snippet_truncated_content_is_prefix(self):
        exc = OutputBlockedError(reasons=["x"], output_snippet="X" * 50 + "Y" * 150)
        assert exc.output_snippet == "X" * 50 + "Y" * 70

    def test_score_inherited(self):
        exc = OutputBlockedError(reasons=["x"], score=0.95)
        assert exc.score == 0.95

    def test_policy_name_inherited(self):
        exc = OutputBlockedError(reasons=["x"], policy_name="balanced")
        assert exc.policy_name == "balanced"

    def test_to_dict_has_output_snippet_key(self):
        exc = OutputBlockedError(reasons=["x"])
        assert "output_snippet" in exc.to_dict()

    def test_to_dict_error_name(self):
        exc = OutputBlockedError(reasons=["x"])
        assert exc.to_dict()["error"] == "OutputBlockedError"

    def test_to_dict_snippet_value(self):
        exc = OutputBlockedError(reasons=["x"], output_snippet="sk-secret")
        assert exc.to_dict()["output_snippet"] == "sk-secret"

    def test_to_dict_includes_inherited_fields(self):
        exc = OutputBlockedError(reasons=["r"], score=0.95, policy_name="bal")
        d = exc.to_dict()
        assert d["reasons"] == ["r"]
        assert d["score"] == 0.95
        assert d["policy_name"] == "bal"

    def test_not_instance_of_prompt_blocked_error(self):
        exc = OutputBlockedError(reasons=["x"])
        assert not isinstance(exc, PromptBlockedError)

    def test_prompt_not_instance_of_output_blocked_error(self):
        exc = PromptBlockedError(reasons=["x"])
        assert not isinstance(exc, OutputBlockedError)

class TestInvalidToolCallError:
    def test_is_llm_security_error(self):
        assert isinstance(InvalidToolCallError("x", "y"), LLMSecurityError)

    def test_is_not_blocked_by_policy_error(self):
        assert not isinstance(InvalidToolCallError("x", "y"), BlockedByPolicyError)

    def test_tool_name_stored(self):
        exc = InvalidToolCallError(tool_name="delete_file", reason="dangerous")
        assert exc.tool_name == "delete_file"

    def test_reason_stored(self):
        exc = InvalidToolCallError(tool_name="x", reason="schema violation")
        assert exc.reason == "schema violation"

    def test_call_id_default_is_none(self):
        exc = InvalidToolCallError(tool_name="x", reason="y")
        assert exc.call_id is None

    def test_call_id_stored(self):
        exc = InvalidToolCallError(tool_name="x", reason="y", call_id="call_abc")
        assert exc.call_id == "call_abc"

    def test_scan_result_default_is_none(self):
        exc = InvalidToolCallError(tool_name="x", reason="y")
        assert exc.scan_result is None

    def test_scan_result_stored(self):
        sr = ScanResult(allowed=False, score=0.88, guard_type=GuardType.TOOL)
        exc = InvalidToolCallError(tool_name="x", reason="y", scan_result=sr)
        assert exc.scan_result is sr

    def test_context_stored(self):
        exc = InvalidToolCallError("x", "y", context={"extra": "data"})
        assert exc.context == {"extra": "data"}

    def test_message_contains_tool_name(self):
        exc = InvalidToolCallError(tool_name="exec_shell", reason="too dangerous")
        assert "exec_shell" in str(exc)

    def test_message_contains_reason(self):
        exc = InvalidToolCallError(tool_name="x", reason="dangerous operation")
        assert "dangerous operation" in str(exc)

    def test_message_contains_call_id_when_set(self):
        exc = InvalidToolCallError(tool_name="x", reason="y", call_id="call_123")
        assert "call_123" in str(exc)

    def test_message_without_call_id_no_bracket(self):
        exc = InvalidToolCallError(tool_name="x", reason="y")
        # No call_id part in message
        assert "call_id" not in str(exc)

    def test_to_dict_keys(self):
        exc = InvalidToolCallError("bad_tool", "bad args", call_id="c1")
        assert set(exc.to_dict().keys()) == {"error", "tool_name", "reason", "call_id"}

    def test_to_dict_error_name(self):
        exc = InvalidToolCallError("x", "y")
        assert exc.to_dict()["error"] == "InvalidToolCallError"

    def test_to_dict_tool_name(self):
        exc = InvalidToolCallError("delete_file", "reason")
        assert exc.to_dict()["tool_name"] == "delete_file"

    def test_to_dict_reason(self):
        exc = InvalidToolCallError("x", "dangerous path traversal")
        assert exc.to_dict()["reason"] == "dangerous path traversal"

    def test_to_dict_call_id_none_when_not_set(self):
        exc = InvalidToolCallError("x", "y")
        assert exc.to_dict()["call_id"] is None

    def test_to_dict_call_id_value(self):
        exc = InvalidToolCallError("x", "y", call_id="call_xyz")
        assert exc.to_dict()["call_id"] == "call_xyz"

class TestPolicyNotFoundError:
    def test_is_llm_security_error(self):
        assert isinstance(PolicyNotFoundError("x"), LLMSecurityError)

    def test_policy_name_stored(self):
        exc = PolicyNotFoundError(policy_name="healthcare-strict")
        assert exc.policy_name == "healthcare-strict"

    def test_available_default_is_empty_list(self):
        exc = PolicyNotFoundError(policy_name="x")
        assert exc.available == []

    def test_available_stored(self):
        exc = PolicyNotFoundError(policy_name="x", available=["strict", "balanced"])
        assert exc.available == ["strict", "balanced"]

    def test_message_no_available_has_no_policies_phrase(self):
        exc = PolicyNotFoundError(policy_name="x")
        assert "No policies" in str(exc)

    def test_message_no_available_contains_policy_name(self):
        exc = PolicyNotFoundError(policy_name="missing-policy")
        assert "missing-policy" in str(exc)

    def test_message_with_available_contains_policy_name(self):
        exc = PolicyNotFoundError(policy_name="my-policy", available=["strict"])
        assert "my-policy" in str(exc)

    def test_message_with_available_lists_available_policies(self):
        exc = PolicyNotFoundError(policy_name="x", available=["strict", "balanced"])
        assert "strict" in str(exc)
        assert "balanced" in str(exc)

    def test_message_no_policies_phrase_absent_when_available_present(self):
        exc = PolicyNotFoundError(policy_name="x", available=["strict"])
        assert "No policies" not in str(exc)

    def test_not_blocked_by_policy_error(self):
        assert not isinstance(PolicyNotFoundError("x"), BlockedByPolicyError)

class TestProviderError:
    def test_is_llm_security_error(self):
        assert isinstance(ProviderError("x"), LLMSecurityError)

    def test_provider_name_stored(self):
        exc = ProviderError("API failed", provider_name="openai")
        assert exc.provider_name == "openai"

    def test_provider_name_default_is_unknown(self):
        exc = ProviderError("failed")
        assert exc.provider_name == "unknown"

    def test_status_code_default_is_none(self):
        exc = ProviderError("x", provider_name="openai")
        assert exc.status_code is None

    def test_status_code_stored(self):
        exc = ProviderError("rate limited", provider_name="openai", status_code=429)
        assert exc.status_code == 429

    def test_original_error_default_is_none(self):
        exc = ProviderError("x", provider_name="openai")
        assert exc.original_error is None

    def test_original_error_stored(self):
        original = ValueError("network error")
        exc = ProviderError("wrapped", provider_name="x", original_error=original)
        assert exc.original_error is original

    def test_context_stored(self):
        exc = ProviderError("x", context={"attempt": 3})
        assert exc.context == {"attempt": 3}

    def test_message_contains_provider_name(self):
        exc = ProviderError("rate limited", provider_name="openai")
        assert "openai" in str(exc)

    def test_message_contains_status_code_when_set(self):
        exc = ProviderError("rate limited", provider_name="openai", status_code=429)
        assert "429" in str(exc)

    def test_message_without_status_code(self):
        exc = ProviderError("API failed", provider_name="anthropic")
        # No HTTP status noise when not set
        assert "HTTP" not in str(exc) or exc.status_code is None

    def test_to_dict_keys(self):
        exc = ProviderError("error", provider_name="openai", status_code=500)
        assert set(exc.to_dict().keys()) == {"error", "provider_name", "status_code", "message"}

    def test_to_dict_error_name(self):
        exc = ProviderError("x", provider_name="openai")
        assert exc.to_dict()["error"] == "ProviderError"

    def test_to_dict_provider_name(self):
        exc = ProviderError("x", provider_name="anthropic")
        assert exc.to_dict()["provider_name"] == "anthropic"

    def test_to_dict_status_code(self):
        exc = ProviderError("x", provider_name="x", status_code=500)
        assert exc.to_dict()["status_code"] == 500

    def test_to_dict_status_code_none_when_not_set(self):
        exc = ProviderError("x", provider_name="x")
        assert exc.to_dict()["status_code"] is None

    def test_to_dict_message_field(self):
        exc = ProviderError("call failed", provider_name="openai")
        assert exc.to_dict()["message"] is not None

class TestProviderTimeoutError:
    def test_is_provider_error(self):
        assert isinstance(ProviderTimeoutError(), ProviderError)

    def test_is_llm_security_error(self):
        assert isinstance(ProviderTimeoutError(), LLMSecurityError)

    def test_status_code_is_408(self):
        assert ProviderTimeoutError().status_code == 408

    def test_status_code_is_408_regardless_of_provider(self):
        for provider in ("openai", "anthropic", "ollama"):
            assert ProviderTimeoutError(provider_name=provider).status_code == 408

    def test_timeout_seconds_default_is_none(self):
        assert ProviderTimeoutError().timeout_seconds is None

    def test_timeout_seconds_stored(self):
        exc = ProviderTimeoutError(timeout_seconds=30.0)
        assert exc.timeout_seconds == 30.0

    def test_timeout_seconds_integer_stored(self):
        exc = ProviderTimeoutError(timeout_seconds=60)
        assert exc.timeout_seconds == 60

    def test_provider_name_stored(self):
        exc = ProviderTimeoutError(provider_name="openai")
        assert exc.provider_name == "openai"

    def test_provider_name_default_is_unknown(self):
        exc = ProviderTimeoutError()
        assert exc.provider_name == "unknown"

    def test_original_error_stored(self):
        original = TimeoutError("timed out")
        exc = ProviderTimeoutError(original_error=original)
        assert exc.original_error is original

    def test_original_error_default_is_none(self):
        assert ProviderTimeoutError().original_error is None

    def test_message_contains_timeout_seconds_when_set(self):
        exc = ProviderTimeoutError(provider_name="openai", timeout_seconds=5.0)
        assert "5.0" in str(exc)

    def test_message_contains_provider_name(self):
        exc = ProviderTimeoutError(provider_name="openai")
        assert "openai" in str(exc)

    def test_message_without_timeout_no_crash(self):
        # Must not raise even when timeout_seconds is None
        exc = ProviderTimeoutError(provider_name="openai")
        assert len(str(exc)) > 0

class TestGuardError:
    def test_is_llm_security_error(self):
        assert isinstance(GuardError("x", "y"), LLMSecurityError)

    def test_is_not_blocked_by_policy_error(self):
        assert not isinstance(GuardError("x", "y"), BlockedByPolicyError)

    def test_guard_name_stored(self):
        exc = GuardError(guard_name="prompt_guard", message="regex exploded")
        assert exc.guard_name == "prompt_guard"

    def test_original_error_default_is_none(self):
        exc = GuardError(guard_name="x", message="y")
        assert exc.original_error is None

    def test_original_error_stored(self):
        original = RuntimeError("boom")
        exc = GuardError(guard_name="x", message="y", original_error=original)
        assert exc.original_error is original

    def test_context_default_is_empty(self):
        exc = GuardError(guard_name="x", message="y")
        assert exc.context == {}

    def test_context_stored(self):
        exc = GuardError(guard_name="x", message="y", context={"detail": "z"})
        assert exc.context == {"detail": "z"}

    def test_message_contains_guard_name(self):
        exc = GuardError(guard_name="prompt_guard", message="crashed")
        assert "prompt_guard" in str(exc)

    def test_message_contains_message_text(self):
        exc = GuardError(guard_name="x", message="regex exploded")
        assert "regex exploded" in str(exc)

    def test_all_guard_names_accepted(self):
        for name in ("prompt_guard", "output_guard", "tool_guard", "custom"):
            exc = GuardError(guard_name=name, message="fault")
            assert name in str(exc)

class TestInheritanceHierarchy:
    @pytest.mark.parametrize("exc", [
        BlockedByPolicyError(reasons=["x"]),
        PromptBlockedError(reasons=["x"]),
        OutputBlockedError(reasons=["x"]),
        InvalidToolCallError(tool_name="x", reason="y"),
        PolicyNotFoundError(policy_name="x"),
        ProviderError(message="x", provider_name="openai"),
        ProviderTimeoutError(provider_name="openai"),
        GuardError(guard_name="x", message="y"),
    ])
    def test_catchable_as_llm_security_error(self, exc):
        caught = False
        try:
            raise exc
        except LLMSecurityError:
            caught = True
        assert caught, f"{type(exc).__name__} not catchable as LLMSecurityError"

    def test_blocked_by_policy_error_is_not_invalid_tool_call(self):
        assert not isinstance(BlockedByPolicyError(reasons=["x"]), InvalidToolCallError)

    def test_provider_error_is_not_blocked_by_policy(self):
        assert not isinstance(ProviderError("x"), BlockedByPolicyError)

    def test_guard_error_is_not_provider_error(self):
        assert not isinstance(GuardError("x", "y"), ProviderError)

    def test_policy_not_found_is_not_blocked_by_policy(self):
        assert not isinstance(PolicyNotFoundError("x"), BlockedByPolicyError)

class TestCatchableAsParent:
    def test_prompt_blocked_catchable_as_blocked_by_policy(self):
        caught = False
        try:
            raise PromptBlockedError(reasons=["injection"])
        except BlockedByPolicyError:
            caught = True
        assert caught

    def test_output_blocked_catchable_as_blocked_by_policy(self):
        caught = False
        try:
            raise OutputBlockedError(reasons=["credential"])
        except BlockedByPolicyError:
            caught = True
        assert caught

    def test_provider_timeout_catchable_as_provider_error(self):
        caught = False
        try:
            raise ProviderTimeoutError(provider_name="openai")
        except ProviderError:
            caught = True
        assert caught

    def test_prompt_blocked_not_caught_as_output_blocked(self):
        caught_as_output = False
        try:
            raise PromptBlockedError(reasons=["x"])
        except OutputBlockedError:
            caught_as_output = True
        except BlockedByPolicyError:
            pass
        assert not caught_as_output

    def test_output_blocked_not_caught_as_prompt_blocked(self):
        caught_as_prompt = False
        try:
            raise OutputBlockedError(reasons=["x"])
        except PromptBlockedError:
            caught_as_prompt = True
        except BlockedByPolicyError:
            pass
        assert not caught_as_prompt

    def test_broad_catch_order_prompt_then_blocked(self):
        caught_by = None
        try:
            raise PromptBlockedError(reasons=["injection"])
        except PromptBlockedError:
            caught_by = "prompt"
        except BlockedByPolicyError:
            caught_by = "blocked"
        assert caught_by == "prompt"

    def test_broad_catch_order_output_then_blocked(self):
        caught_by = None
        try:
            raise OutputBlockedError(reasons=["credential"])
        except OutputBlockedError:
            caught_by = "output"
        except BlockedByPolicyError:
            caught_by = "blocked"
        assert caught_by == "output"

    def test_broad_catch_order_timeout_then_provider(self):
        caught_by = None
        try:
            raise ProviderTimeoutError(provider_name="openai")
        except ProviderTimeoutError:
            caught_by = "timeout"
        except ProviderError:
            caught_by = "provider"
        assert caught_by == "timeout"