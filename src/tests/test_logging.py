from __future__ import annotations
import json
import logging as stdlib_logging
from typing import List
import pytest
from src.logging import (
    EventType,
    LogFormat,
    SecurityEvent,
    SecurityEventLogger,
    add_handler,
    clear_handlers,
    log_decision,
    log_scan_result,
    log_tool_call,
    remove_handler,
)
from src.policies import BalancedPolicy
from src.types import (
    GuardDecision,
    GuardType,
    ScanResult,
    ToolCall,
)

@pytest.fixture(autouse=True)
def _clean_handlers():
    clear_handlers()
    yield
    clear_handlers()

@pytest.fixture()
def policy():
    return BalancedPolicy()

@pytest.fixture()
def clean_decision():
    return GuardDecision.clean("The capital is Paris.")

@pytest.fixture()
def blocked_decision():
    return GuardDecision.blocked(["injection detected"], score=0.92)

@pytest.fixture()
def warned_decision():
    return GuardDecision.allowed_with_warning("safe text", ["credential noted"], 0.45)

@pytest.fixture()
def clean_scan():
    return ScanResult(allowed=True, score=0.0, guard_type=GuardType.PROMPT)

@pytest.fixture()
def blocked_scan():
    return ScanResult(
        allowed=False,
        score=0.92,
        reasons=["injection"],
        guard_type=GuardType.PROMPT,
    )

@pytest.fixture()
def safe_tool_call():
    return ToolCall(name="search_web", args={"query": "python"}, call_id="call_abc")

@pytest.fixture()
def captured_events() -> List[SecurityEvent]:
    events: List[SecurityEvent] = []
    add_handler(events.append)
    return events

class TestLogFormatEnum:
    def test_json_value(self):
        assert LogFormat.JSON.value == "json"

    def test_text_value(self):
        assert LogFormat.TEXT.value == "text"

    def test_is_str_subclass(self):
        assert LogFormat.JSON == "json"
        assert LogFormat.TEXT == "text"

class TestEventTypeEnum:
    def test_decision_value(self):
        assert EventType.DECISION.value == "decision"

    def test_scan_value(self):
        assert EventType.SCAN.value == "scan"

    def test_tool_call_value(self):
        assert EventType.TOOL_CALL.value == "tool_call"

    def test_is_str_subclass(self):
        assert EventType.DECISION == "decision"
        assert EventType.SCAN == "scan"
        assert EventType.TOOL_CALL == "tool_call"

class TestSecurityEventConstruction:
    def _make(self, **kwargs):
        defaults = dict(
            event_type=EventType.DECISION,
            timestamp="2025-03-09T14:23:01.456789+00:00",
            allowed=True,
            score=0.0,
            risk_level="none",
            reasons=[],
            action="log",
        )
        defaults.update(kwargs)
        return SecurityEvent(**defaults)

    def test_event_type_stored(self):
        assert self._make(event_type=EventType.SCAN).event_type == EventType.SCAN

    def test_timestamp_stored(self):
        ts = "2025-03-09T14:23:01.456789+00:00"
        assert self._make(timestamp=ts).timestamp == ts

    def test_allowed_stored(self):
        assert self._make(allowed=False).allowed is False

    def test_score_stored(self):
        assert self._make(score=0.92).score == 0.92

    def test_risk_level_stored(self):
        assert self._make(risk_level="critical").risk_level == "critical"

    def test_reasons_stored(self):
        assert self._make(reasons=["r1", "r2"]).reasons == ["r1", "r2"]

    def test_action_stored(self):
        assert self._make(action="block").action == "block"

    def test_guard_type_defaults_none(self):
        assert self._make().guard_type is None

    def test_policy_name_defaults_unknown(self):
        assert self._make().policy_name == "unknown"

    def test_provider_name_defaults_unknown(self):
        assert self._make().provider_name == "unknown"

    def test_tool_name_defaults_none(self):
        assert self._make().tool_name is None

    def test_tool_call_id_defaults_none(self):
        assert self._make().tool_call_id is None

    def test_safe_output_present_defaults_false(self):
        assert self._make().safe_output_present is False

    def test_duration_ms_defaults_none(self):
        assert self._make().duration_ms is None

    def test_trace_id_defaults_none(self):
        assert self._make().trace_id is None

    def test_span_id_defaults_none(self):
        assert self._make().span_id is None

    def test_extra_defaults_empty_dict(self):
        assert self._make().extra == {}

    def test_extra_not_shared_between_instances(self):
        e1 = self._make()
        e2 = self._make()
        e1.extra["k"] = "v"
        assert "k" not in e2.extra

class TestSecurityEventToDict:
    EXPECTED_KEYS = {
        "event_type", "timestamp", "allowed", "score", "risk_level",
        "reasons", "action", "guard_type", "policy_name", "provider_name",
        "tool_name", "tool_call_id", "safe_output_present", "duration_ms",
        "trace_id", "span_id", "extra",
    }

    def _make(self):
        return SecurityEvent(
            event_type=EventType.DECISION,
            timestamp="2025-03-09T14:23:01.456789+00:00",
            allowed=True, score=0.0, risk_level="none",
            reasons=[], action="log",
        )

    def test_contains_all_expected_keys(self):
        assert set(self._make().to_dict().keys()) == self.EXPECTED_KEYS

    def test_event_type_is_string(self):
        assert isinstance(self._make().to_dict()["event_type"], str)

    def test_event_type_value_decision(self):
        assert self._make().to_dict()["event_type"] == "decision"

    def test_scan_event_type_value(self):
        evt = SecurityEvent(
            event_type=EventType.SCAN, timestamp="x",
            allowed=True, score=0.0, risk_level="none", reasons=[], action="log",
        )
        assert evt.to_dict()["event_type"] == "scan"

    def test_tool_call_event_type_value(self):
        evt = SecurityEvent(
            event_type=EventType.TOOL_CALL, timestamp="x",
            allowed=True, score=0.0, risk_level="none", reasons=[], action="log",
        )
        assert evt.to_dict()["event_type"] == "tool_call"

    def test_allowed_field(self):
        assert self._make().to_dict()["allowed"] is True

    def test_optional_none_fields_present_as_none(self):
        d = self._make().to_dict()
        assert d["guard_type"] is None
        assert d["tool_name"] is None
        assert d["duration_ms"] is None
        assert d["trace_id"] is None
        assert d["span_id"] is None

    def test_extra_preserved(self):
        evt = SecurityEvent(
            event_type=EventType.DECISION, timestamp="x",
            allowed=True, score=0.0, risk_level="none",
            reasons=[], action="log", extra={"req_id": "r1"},
        )
        assert evt.to_dict()["extra"] == {"req_id": "r1"}

class TestSecurityEventToJson:
    def _make(self):
        return SecurityEvent(
            event_type=EventType.DECISION,
            timestamp="2025-03-09T14:23:01.456789+00:00",
            allowed=True, score=0.0, risk_level="none",
            reasons=[], action="log",
        )

    def test_returns_string(self):
        assert isinstance(self._make().to_json(), str)

    def test_is_valid_json(self):
        parsed = json.loads(self._make().to_json())
        assert isinstance(parsed, dict)

    def test_event_type_in_json_is_string(self):
        assert json.loads(self._make().to_json())["event_type"] == "decision"

    def test_all_keys_present_in_json(self):
        parsed = json.loads(self._make().to_json())
        assert "allowed" in parsed
        assert "score" in parsed
        assert "timestamp" in parsed

    def test_single_line_no_newline(self):
        assert "\n" not in self._make().to_json()

class TestSecurityEventToText:
    @pytest.fixture()
    def full_event(self):
        return SecurityEvent(
            event_type=EventType.DECISION,
            timestamp="2025-03-09T14:23:01.456789+00:00",
            allowed=False, score=0.92, risk_level="critical",
            reasons=["injection detected"], action="block",
            policy_name="balanced", provider_name="openai",
            guard_type="prompt", tool_name="my_tool",
            tool_call_id="call_123", duration_ms=12.5,
            trace_id="trace-abc",
        )

    def test_contains_event_type(self, full_event):
        assert "event=decision" in full_event.to_text()

    def test_contains_allowed_false(self, full_event):
        assert "allowed=False" in full_event.to_text()

    def test_contains_score(self, full_event):
        assert "score=0.9200" in full_event.to_text()

    def test_contains_risk(self, full_event):
        assert "risk=critical" in full_event.to_text()

    def test_contains_action(self, full_event):
        assert "action=block" in full_event.to_text()

    def test_contains_policy(self, full_event):
        assert "policy=balanced" in full_event.to_text()

    def test_contains_provider(self, full_event):
        assert "provider=openai" in full_event.to_text()

    def test_contains_guard(self, full_event):
        assert "guard=prompt" in full_event.to_text()

    def test_contains_tool(self, full_event):
        assert "tool=my_tool" in full_event.to_text()

    def test_contains_call_id(self, full_event):
        assert "call_id=call_123" in full_event.to_text()

    def test_contains_duration_ms(self, full_event):
        assert "duration_ms=12.5" in full_event.to_text()

    def test_contains_trace_id(self, full_event):
        assert "trace_id=trace-abc" in full_event.to_text()

    def test_contains_reasons(self, full_event):
        assert "reasons=" in full_event.to_text()
        assert "injection detected" in full_event.to_text()

    def test_returns_single_line(self, full_event):
        assert "\n" not in full_event.to_text()

    def test_extra_key_value_included(self):
        evt = SecurityEvent(
            event_type=EventType.DECISION,
            timestamp="2025-03-09T14:23:01.456789+00:00",
            allowed=True, score=0.0, risk_level="none",
            reasons=[], action="log", extra={"req_id": "r123"},
        )
        assert "req_id" in evt.to_text()

class TestSecurityEventToTextClean:
    @pytest.fixture()
    def clean_event(self):
        return SecurityEvent(
            event_type=EventType.DECISION,
            timestamp="2025-03-09T14:23:01.456789+00:00",
            allowed=True, score=0.0, risk_level="none",
            reasons=[], action="log",
        )

    def test_no_guard_field_when_none(self, clean_event):
        assert "guard=" not in clean_event.to_text()

    def test_no_tool_field_when_none(self, clean_event):
        assert "tool=" not in clean_event.to_text()

    def test_no_call_id_field_when_none(self, clean_event):
        assert "call_id=" not in clean_event.to_text()

    def test_no_reasons_field_when_empty(self, clean_event):
        assert "reasons=" not in clean_event.to_text()

    def test_no_duration_ms_when_none(self, clean_event):
        assert "duration_ms" not in clean_event.to_text()

    def test_no_trace_id_when_none(self, clean_event):
        assert "trace_id" not in clean_event.to_text()

    def test_allowed_true_in_output(self, clean_event):
        assert "allowed=True" in clean_event.to_text()

class TestHandlerRegistry:
    def test_add_handler_receives_events(self, policy, clean_decision, captured_events):
        log_decision(clean_decision, policy=policy)
        assert len(captured_events) == 1

    def test_handler_receives_security_event_instance(self, policy, clean_decision, captured_events):
        log_decision(clean_decision, policy=policy)
        assert isinstance(captured_events[0], SecurityEvent)

    def test_multiple_handlers_all_called(self, policy, clean_decision):
        a, b = [], []
        add_handler(a.append)
        add_handler(b.append)
        log_decision(clean_decision, policy=policy)
        assert len(a) == 1 and len(b) == 1

    def test_handlers_called_in_registration_order(self, policy, clean_decision):
        order = []
        add_handler(lambda e: order.append("first"))
        add_handler(lambda e: order.append("second"))
        log_decision(clean_decision, policy=policy)
        assert order == ["first", "second"]

    def test_remove_handler_stops_dispatch(self, policy, clean_decision):
        events = []
        fn = events.append
        add_handler(fn)
        remove_handler(fn)
        log_decision(clean_decision, policy=policy)
        assert len(events) == 0

    def test_remove_non_existent_handler_is_silent(self):
        remove_handler(lambda e: None)  # must not raise

    def test_clear_handlers_removes_all(self, policy, clean_decision, captured_events):
        clear_handlers()
        log_decision(clean_decision, policy=policy)
        assert len(captured_events) == 0

    def test_multiple_log_calls_accumulate_events(
        self, policy, clean_decision, blocked_decision, captured_events
    ):
        log_decision(clean_decision, policy=policy)
        log_decision(blocked_decision, policy=policy)
        assert len(captured_events) == 2

    def test_remove_one_of_two_handlers(self, policy, clean_decision):
        e1, e2 = [], []
        fn1 = e1.append
        add_handler(fn1)
        add_handler(e2.append)
        remove_handler(fn1)
        log_decision(clean_decision, policy=policy)
        assert len(e1) == 0 and len(e2) == 1

class TestHandlerFailureIsolation:
    def test_exception_in_handler_does_not_propagate(self, policy, clean_decision):
        add_handler(lambda e: (_ for _ in ()).throw(RuntimeError("crash")))
        log_decision(clean_decision, policy=policy)  # must not raise

    def test_subsequent_handler_still_called_after_crash(self, policy, clean_decision):
        good = []
        def bad(e): raise ValueError("oops")
        add_handler(bad)
        add_handler(good.append)
        log_decision(clean_decision, policy=policy)
        assert len(good) == 1

    def test_all_subsequent_handlers_called_after_crash(self, policy, clean_decision):
        counts = []
        def bad(e): raise RuntimeError("x")
        add_handler(bad)
        add_handler(lambda e: counts.append(1))
        add_handler(lambda e: counts.append(2))
        log_decision(clean_decision, policy=policy)
        assert counts == [1, 2]

class TestLogDecision:
    def test_returns_security_event(self, policy, clean_decision):
        assert isinstance(log_decision(clean_decision, policy=policy), SecurityEvent)

    def test_event_type_is_decision(self, policy, clean_decision, captured_events):
        log_decision(clean_decision, policy=policy)
        assert captured_events[0].event_type == EventType.DECISION

    def test_allowed_true_for_clean(self, policy, clean_decision, captured_events):
        log_decision(clean_decision, policy=policy)
        assert captured_events[0].allowed is True

    def test_allowed_false_for_blocked(self, policy, blocked_decision, captured_events):
        log_decision(blocked_decision, policy=policy)
        assert captured_events[0].allowed is False

    def test_score_from_decision(self, policy, blocked_decision, captured_events):
        log_decision(blocked_decision, policy=policy)
        assert captured_events[0].score == 0.92

    def test_reasons_from_decision(self, policy, blocked_decision, captured_events):
        log_decision(blocked_decision, policy=policy)
        assert captured_events[0].reasons == ["injection detected"]

    def test_action_block_for_blocked(self, policy, blocked_decision, captured_events):
        log_decision(blocked_decision, policy=policy)
        assert captured_events[0].action == "block"

    def test_action_log_for_clean(self, policy, clean_decision, captured_events):
        log_decision(clean_decision, policy=policy)
        assert captured_events[0].action == "log"

    def test_policy_name_from_policy(self, policy, clean_decision, captured_events):
        log_decision(clean_decision, policy=policy)
        assert captured_events[0].policy_name == "balanced"

    def test_provider_name_set(self, policy, clean_decision, captured_events):
        log_decision(clean_decision, policy=policy, provider_name="openai")
        assert captured_events[0].provider_name == "openai"

    def test_risk_level_none_for_clean(self, policy, clean_decision, captured_events):
        log_decision(clean_decision, policy=policy)
        assert captured_events[0].risk_level == "none"

    def test_risk_level_critical_for_high_score(self, policy, blocked_decision, captured_events):
        log_decision(blocked_decision, policy=policy)
        assert captured_events[0].risk_level == "critical"

    def test_timestamp_is_iso8601_utc(self, policy, clean_decision, captured_events):
        log_decision(clean_decision, policy=policy)
        ts = captured_events[0].timestamp
        assert "T" in ts
        assert "+" in ts or "Z" in ts

    def test_no_policy_defaults_policy_name_unknown(self, clean_decision, captured_events):
        log_decision(clean_decision)
        assert captured_events[0].policy_name == "unknown"

    def test_no_provider_defaults_unknown(self, clean_decision, captured_events):
        log_decision(clean_decision)
        assert captured_events[0].provider_name == "unknown"

class TestLogDecisionSafeOutputFlag:
    def test_true_when_safe_output_set(self, policy, captured_events):
        log_decision(GuardDecision.clean("some output"), policy=policy)
        assert captured_events[0].safe_output_present is True

    def test_false_when_blocked(self, policy, captured_events):
        log_decision(GuardDecision.blocked(["x"]), policy=policy)
        assert captured_events[0].safe_output_present is False

    def test_false_when_clean_with_none_output(self, policy, captured_events):
        log_decision(GuardDecision.clean(None), policy=policy)
        assert captured_events[0].safe_output_present is False

    def test_true_when_warned_with_output(self, policy, captured_events):
        d = GuardDecision.allowed_with_warning("response", ["r"], 0.45)
        log_decision(d, policy=policy)
        assert captured_events[0].safe_output_present is True

class TestLogDecisionOptionalFields:
    def test_duration_ms_forwarded(self, policy, clean_decision, captured_events):
        log_decision(clean_decision, policy=policy, duration_ms=42.5)
        assert captured_events[0].duration_ms == 42.5

    def test_duration_ms_default_none(self, policy, clean_decision, captured_events):
        log_decision(clean_decision, policy=policy)
        assert captured_events[0].duration_ms is None

    def test_trace_id_forwarded(self, policy, clean_decision, captured_events):
        log_decision(clean_decision, policy=policy, trace_id="t-abc")
        assert captured_events[0].trace_id == "t-abc"

    def test_span_id_forwarded(self, policy, clean_decision, captured_events):
        log_decision(clean_decision, policy=policy, span_id="s-xyz")
        assert captured_events[0].span_id == "s-xyz"

    def test_trace_and_span_default_none(self, policy, clean_decision, captured_events):
        log_decision(clean_decision, policy=policy)
        assert captured_events[0].trace_id is None
        assert captured_events[0].span_id is None

    def test_extra_forwarded(self, policy, clean_decision, captured_events):
        log_decision(clean_decision, policy=policy, extra={"req_id": "r123"})
        assert captured_events[0].extra == {"req_id": "r123"}

    def test_extra_default_empty_dict(self, policy, clean_decision, captured_events):
        log_decision(clean_decision, policy=policy)
        assert captured_events[0].extra == {}

class TestLogScanResult:
    def test_returns_security_event(self, policy, clean_scan):
        assert isinstance(log_scan_result(clean_scan, policy=policy), SecurityEvent)

    def test_event_type_is_scan(self, policy, clean_scan, captured_events):
        log_scan_result(clean_scan, policy=policy)
        assert captured_events[0].event_type == EventType.SCAN

    def test_guard_type_prompt(self, policy, clean_scan, captured_events):
        log_scan_result(clean_scan, policy=policy)
        assert captured_events[0].guard_type == "prompt"

    def test_allowed_from_scan_result(self, policy, clean_scan, captured_events):
        log_scan_result(clean_scan, policy=policy)
        assert captured_events[0].allowed is True

    def test_blocked_scan_event(self, policy, blocked_scan, captured_events):
        log_scan_result(blocked_scan, policy=policy)
        evt = captured_events[0]
        assert evt.allowed is False
        assert evt.score == 0.92
        assert evt.reasons == ["injection"]
        assert evt.action == "block"

    def test_clean_scan_action_is_log(self, policy, clean_scan, captured_events):
        log_scan_result(clean_scan, policy=policy)
        assert captured_events[0].action == "log"

    def test_output_guard_type(self, policy, captured_events):
        sr = ScanResult(allowed=True, score=0.0, guard_type=GuardType.OUTPUT)
        log_scan_result(sr, policy=policy)
        assert captured_events[0].guard_type == "output"

    def test_tool_guard_type(self, policy, captured_events):
        sr = ScanResult(allowed=True, score=0.0, guard_type=GuardType.TOOL)
        log_scan_result(sr, policy=policy)
        assert captured_events[0].guard_type == "tool"

    def test_policy_name_attached(self, policy, clean_scan, captured_events):
        log_scan_result(clean_scan, policy=policy)
        assert captured_events[0].policy_name == "balanced"

    def test_provider_name_attached(self, policy, clean_scan, captured_events):
        log_scan_result(clean_scan, policy=policy, provider_name="anthropic")
        assert captured_events[0].provider_name == "anthropic"

    def test_extra_forwarded(self, policy, clean_scan, captured_events):
        log_scan_result(clean_scan, policy=policy, extra={"env": "prod"})
        assert captured_events[0].extra == {"env": "prod"}

    @pytest.mark.parametrize("guard_type", list(GuardType))
    def test_all_guard_types_produce_correct_field(self, policy, captured_events, guard_type):
        sr = ScanResult(allowed=True, score=0.0, guard_type=guard_type)
        log_scan_result(sr, policy=policy)
        assert captured_events[-1].guard_type == guard_type.value

class TestLogToolCall:
    def test_returns_security_event(self, policy, safe_tool_call):
        assert isinstance(log_tool_call(safe_tool_call, policy=policy), SecurityEvent)

    def test_event_type_is_tool_call(self, policy, safe_tool_call, captured_events):
        log_tool_call(safe_tool_call, policy=policy)
        assert captured_events[0].event_type == EventType.TOOL_CALL

    def test_tool_name_set(self, policy, safe_tool_call, captured_events):
        log_tool_call(safe_tool_call, policy=policy)
        assert captured_events[0].tool_name == "search_web"

    def test_tool_call_id_set(self, policy, safe_tool_call, captured_events):
        log_tool_call(safe_tool_call, policy=policy)
        assert captured_events[0].tool_call_id == "call_abc"

    def test_guard_type_is_tool(self, policy, safe_tool_call, captured_events):
        log_tool_call(safe_tool_call, policy=policy)
        assert captured_events[0].guard_type == "tool"

    def test_without_result_allowed_true(self, policy, safe_tool_call, captured_events):
        log_tool_call(safe_tool_call, policy=policy)
        assert captured_events[0].allowed is True

    def test_without_result_score_zero(self, policy, safe_tool_call, captured_events):
        log_tool_call(safe_tool_call, policy=policy)
        assert captured_events[0].score == 0.0

    def test_without_result_action_log(self, policy, safe_tool_call, captured_events):
        log_tool_call(safe_tool_call, policy=policy)
        assert captured_events[0].action == "log"

    def test_with_blocked_result(self, policy, safe_tool_call, captured_events):
        sr = ScanResult(
            allowed=False, score=0.88,
            reasons=["dangerous"], guard_type=GuardType.TOOL,
        )
        log_tool_call(safe_tool_call, sr, policy=policy)
        evt = captured_events[0]
        assert evt.allowed is False
        assert evt.score == 0.88
        assert evt.reasons == ["dangerous"]
        assert evt.action == "block"

    def test_without_call_id(self, policy, captured_events):
        tc = ToolCall(name="ping", args={})
        log_tool_call(tc, policy=policy)
        assert captured_events[0].tool_call_id is None

    def test_policy_name_attached(self, policy, safe_tool_call, captured_events):
        log_tool_call(safe_tool_call, policy=policy)
        assert captured_events[0].policy_name == "balanced"

    def test_provider_name_attached(self, policy, safe_tool_call, captured_events):
        log_tool_call(safe_tool_call, policy=policy, provider_name="openai")
        assert captured_events[0].provider_name == "openai"

    def test_extra_forwarded(self, policy, safe_tool_call, captured_events):
        log_tool_call(safe_tool_call, policy=policy, extra={"env": "test"})
        assert captured_events[0].extra == {"env": "test"}

class TestStdlibLoggingLevel:
    class _Cap(stdlib_logging.Handler):
        def __init__(self):
            super().__init__()
            self.records = []
        def emit(self, record):
            self.records.append(record)

    def _attach(self):
        import src.logging as llm_log
        cap = self._Cap()
        llm_log._internal_logger.addHandler(cap)
        llm_log._internal_logger.setLevel(stdlib_logging.DEBUG)
        return cap, llm_log

    def _detach(self, cap, llm_log):
        llm_log._internal_logger.removeHandler(cap)

    def test_blocked_logs_at_error(self):
        cap, llm_log = self._attach()
        try:
            log_decision(GuardDecision.blocked(["x"]), policy=BalancedPolicy())
            assert cap.records[-1].levelno == stdlib_logging.ERROR
        finally:
            self._detach(cap, llm_log)

    def test_warned_logs_at_warning(self):
        cap, llm_log = self._attach()
        try:
            d = GuardDecision.allowed_with_warning("x", ["r"], 0.45)
            log_decision(d, policy=BalancedPolicy())
            assert cap.records[-1].levelno == stdlib_logging.WARNING
        finally:
            self._detach(cap, llm_log)

    def test_clean_logs_at_info(self):
        cap, llm_log = self._attach()
        try:
            log_decision(GuardDecision.clean("hi"), policy=BalancedPolicy())
            assert cap.records[-1].levelno == stdlib_logging.INFO
        finally:
            self._detach(cap, llm_log)

class TestSecurityEventLogger:
    def test_policy_stored(self, policy):
        assert SecurityEventLogger(policy=policy).policy is policy

    def test_provider_name_stored(self):
        assert SecurityEventLogger(provider_name="anthropic").provider_name == "anthropic"

    def test_fmt_defaults_to_json(self):
        assert SecurityEventLogger().fmt == LogFormat.JSON

    def test_fmt_text_accepted(self):
        assert SecurityEventLogger(fmt=LogFormat.TEXT).fmt == LogFormat.TEXT

    def test_default_extra_defaults_empty(self):
        assert SecurityEventLogger().default_extra == {}

    def test_log_decision_attaches_provider_name(self, policy, captured_events):
        SecurityEventLogger(policy=policy, provider_name="openai").log_decision(
            GuardDecision.clean("hi")
        )
        assert captured_events[0].provider_name == "openai"

    def test_log_decision_attaches_policy_name(self, policy, captured_events):
        SecurityEventLogger(policy=policy, provider_name="x").log_decision(
            GuardDecision.clean("hi")
        )
        assert captured_events[0].policy_name == "balanced"

    def test_log_decision_returns_security_event(self, policy):
        logger = SecurityEventLogger(policy=policy, provider_name="x")
        assert isinstance(logger.log_decision(GuardDecision.clean("hi")), SecurityEvent)

    def test_log_decision_forwards_duration_ms(self, policy, captured_events):
        SecurityEventLogger(policy=policy, provider_name="x").log_decision(
            GuardDecision.clean("hi"), duration_ms=15.0
        )
        assert captured_events[0].duration_ms == 15.0

    def test_log_decision_forwards_trace_id(self, policy, captured_events):
        SecurityEventLogger(policy=policy, provider_name="x").log_decision(
            GuardDecision.clean("hi"), trace_id="t1"
        )
        assert captured_events[0].trace_id == "t1"

    def test_log_decision_forwards_span_id(self, policy, captured_events):
        SecurityEventLogger(policy=policy, provider_name="x").log_decision(
            GuardDecision.clean("hi"), span_id="s1"
        )
        assert captured_events[0].span_id == "s1"

    def test_log_scan_result_delegates(self, policy, captured_events):
        logger = SecurityEventLogger(policy=policy, provider_name="openai")
        sr = ScanResult(allowed=True, score=0.0, guard_type=GuardType.PROMPT)
        result = logger.log_scan_result(sr)
        assert isinstance(result, SecurityEvent)
        assert captured_events[0].event_type == EventType.SCAN
        assert captured_events[0].provider_name == "openai"

    def test_log_tool_call_delegates(self, policy, captured_events):
        logger = SecurityEventLogger(policy=policy, provider_name="openai")
        tc = ToolCall(name="search", args={})
        result = logger.log_tool_call(tc)
        assert isinstance(result, SecurityEvent)
        assert captured_events[0].event_type == EventType.TOOL_CALL
        assert captured_events[0].provider_name == "openai"
        assert captured_events[0].tool_name == "search"

    def test_no_policy_defaults_to_unknown(self, captured_events):
        SecurityEventLogger(policy=None, provider_name="x").log_decision(
            GuardDecision.clean("hi")
        )
        assert captured_events[0].policy_name == "unknown"

class TestSecurityEventLoggerExtra:
    def test_default_extra_in_every_event(self, policy, captured_events):
        logger = SecurityEventLogger(
            policy=policy, provider_name="x",
            default_extra={"service": "chat-api"},
        )
        logger.log_decision(GuardDecision.clean("hi"))
        assert captured_events[0].extra.get("service") == "chat-api"

    def test_per_call_extra_merged_with_default(self, policy, captured_events):
        logger = SecurityEventLogger(
            policy=policy, provider_name="x",
            default_extra={"service": "chat-api"},
        )
        logger.log_decision(GuardDecision.clean("hi"), extra={"req_id": "r1"})
        evt = captured_events[0]
        assert evt.extra.get("service") == "chat-api"
        assert evt.extra.get("req_id") == "r1"

    def test_per_call_extra_overrides_default(self, policy, captured_events):
        logger = SecurityEventLogger(
            policy=policy, provider_name="x",
            default_extra={"service": "chat-api"},
        )
        logger.log_decision(GuardDecision.clean("hi"), extra={"service": "override"})
        assert captured_events[0].extra.get("service") == "override"

    def test_default_extra_not_mutated(self, policy):
        logger = SecurityEventLogger(
            policy=policy, provider_name="x",
            default_extra={"service": "chat-api"},
        )
        logger.log_decision(GuardDecision.clean("hi"), extra={"req_id": "r1"})
        assert "req_id" not in logger.default_extra

    def test_no_extras_gives_empty_dict(self, policy, captured_events):
        SecurityEventLogger(policy=policy, provider_name="x").log_decision(
            GuardDecision.clean("hi")
        )
        assert captured_events[0].extra == {}

    def test_default_extra_on_log_scan_result(self, policy, captured_events):
        logger = SecurityEventLogger(
            policy=policy, provider_name="x",
            default_extra={"env": "staging"},
        )
        sr = ScanResult(allowed=True, score=0.0, guard_type=GuardType.PROMPT)
        logger.log_scan_result(sr)
        assert captured_events[0].extra.get("env") == "staging"

    def test_default_extra_on_log_tool_call(self, policy, captured_events):
        logger = SecurityEventLogger(
            policy=policy, provider_name="x",
            default_extra={"env": "staging"},
        )
        logger.log_tool_call(ToolCall(name="search", args={}))
        assert captured_events[0].extra.get("env") == "staging"