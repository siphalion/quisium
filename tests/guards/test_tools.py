from __future__ import annotations
import pytest
from quisium.guards.tools import (
    aggregate_tool_scans,
    validate_tool_call,
    validate_tool_calls,
)
from quisium.policies import BalancedPolicy, GuardConfig, LoggingOnlyPolicy, StrictPolicy
from quisium.types import GuardType, ScanResult, ToolCall

@pytest.fixture()
def balanced():
    return BalancedPolicy(raise_on_block=False)   # block=0.75

@pytest.fixture()
def strict():
    return StrictPolicy(raise_on_block=False)     # block=0.40

@pytest.fixture()
def logging_only():
    return LoggingOnlyPolicy()                    # block=1.0, never blocks

@pytest.fixture()
def disabled_policy():
    return BalancedPolicy(
        tool_guard=GuardConfig(enabled=False),
        raise_on_block=False,
    )

def tc(name: str, args: dict, schema: dict | None = None, call_id: str | None = None) -> ToolCall:
    return ToolCall(name=name, args=args, schema=schema or {}, call_id=call_id)

_FULL_SCHEMA = {
    "type": "object",
    "properties": {
        "username": {"type": "string"},
        "age":      {"type": "integer"},
        "limit":    {"type": "integer", "minimum": 1, "maximum": 100},
    },
    "required": ["username", "age"],
}

class TestValidateToolCallSafe:
    def test_web_search_is_safe(self, balanced):
        r = validate_tool_call(tc("search_web", {"query": "python tutorial"}), balanced)
        assert r.allowed is True
        assert r.score == 0.0

    def test_get_weather_is_safe(self, balanced):
        r = validate_tool_call(tc("get_weather", {"city": "London", "units": "metric"}), balanced)
        assert r.allowed is True
        assert r.score == 0.0

    def test_read_file_safe_path_is_safe(self, balanced):
        r = validate_tool_call(tc("read_file", {"path": "documents/report.pdf"}), balanced)
        assert r.allowed is True
        assert r.score == 0.0

    def test_send_email_is_safe(self, balanced):
        r = validate_tool_call(tc("send_email", {"to": "user@example.com", "subject": "Hello"}), balanced)
        assert r.allowed is True

    def test_safe_call_has_empty_reasons(self, balanced):
        r = validate_tool_call(tc("search_web", {"query": "hello"}), balanced)
        assert r.reasons == []

    def test_safe_call_has_empty_categories(self, balanced):
        r = validate_tool_call(tc("search_web", {"query": "hello"}), balanced)
        assert r.metadata.get("categories") == []

    def test_safe_call_returns_scan_result(self, balanced):
        r = validate_tool_call(tc("search_web", {"query": "hello"}), balanced)
        assert isinstance(r, ScanResult)

    def test_guard_type_is_tool(self, balanced):
        r = validate_tool_call(tc("search_web", {"query": "hello"}), balanced)
        assert r.guard_type == GuardType.TOOL

    def test_empty_args_is_safe(self, balanced):
        r = validate_tool_call(tc("ping", {}), balanced)
        assert r.allowed is True

class TestValidateToolCallDenylist:
    @pytest.fixture()
    def deny_policy(self):
        return BalancedPolicy(
            blocked_tools={"exec_shell", "delete_all"},
            raise_on_block=False,
        )

    def test_blocked_tool_not_allowed(self, deny_policy):
        r = validate_tool_call(tc("exec_shell", {}), deny_policy)
        assert r.allowed is False

    def test_blocked_tool_score_is_one(self, deny_policy):
        r = validate_tool_call(tc("exec_shell", {}), deny_policy)
        assert r.score == 1.0

    def test_blocked_tool_category_denylist_violation(self, deny_policy):
        r = validate_tool_call(tc("exec_shell", {}), deny_policy)
        assert "denylist_violation" in r.metadata.get("categories", [])

    def test_second_blocked_tool(self, deny_policy):
        r = validate_tool_call(tc("delete_all", {}), deny_policy)
        assert r.allowed is False
        assert r.score == 1.0

    def test_non_blocked_tool_passes(self, deny_policy):
        r = validate_tool_call(tc("search_web", {"query": "hello"}), deny_policy)
        assert r.allowed is True

    def test_blocked_tool_reason_mentions_tool_name(self, deny_policy):
        r = validate_tool_call(tc("exec_shell", {}), deny_policy)
        assert any("exec_shell" in reason for reason in r.reasons)

class TestValidateToolCallAllowlist:
    @pytest.fixture()
    def allow_policy(self):
        return BalancedPolicy(
            allowed_tools=["search_web", "read_file"],
            raise_on_block=False,
        )

    def test_unlisted_tool_not_allowed(self, allow_policy):
        r = validate_tool_call(tc("delete_file", {}), allow_policy)
        assert r.allowed is False

    def test_unlisted_tool_score(self, allow_policy):
        r = validate_tool_call(tc("delete_file", {}), allow_policy)
        assert r.score == 0.95

    def test_unlisted_tool_category(self, allow_policy):
        r = validate_tool_call(tc("delete_file", {}), allow_policy)
        assert "allowlist_violation" in r.metadata.get("categories", [])

    def test_listed_tool_passes(self, allow_policy):
        r = validate_tool_call(tc("search_web", {"query": "hello"}), allow_policy)
        assert r.allowed is True

    def test_second_listed_tool_passes(self, allow_policy):
        r = validate_tool_call(tc("read_file", {"path": "documents/report.pdf"}), allow_policy)
        assert r.allowed is True

    def test_allowlist_violation_reason_mentions_permitted(self, allow_policy):
        r = validate_tool_call(tc("exec_shell", {}), allow_policy)
        assert any("search_web" in reason or "read_file" in reason for reason in r.reasons)

class TestValidateToolCallDangerousNames:
    @pytest.mark.parametrize("name", [
        "exec", "eval", "shell", "run_command", "execute_command",
        "delete_all", "format_disk", "drop_database", "wipe",
    ])
    def test_dangerous_name_blocked(self, balanced, name):
        r = validate_tool_call(tc(name, {}), balanced)
        assert r.allowed is False
        assert r.score == 0.95

    def test_dangerous_name_category(self, balanced):
        r = validate_tool_call(tc("exec", {}), balanced)
        assert "dangerous_operation" in r.metadata.get("categories", [])

    def test_dangerous_name_reason_contains_name(self, balanced):
        r = validate_tool_call(tc("shell", {}), balanced)
        assert any("shell" in reason for reason in r.reasons)

class TestValidateToolCallAdminNames:
    @pytest.mark.parametrize("name", [
        "admin_panel", "superuser_create", "debug_endpoint",
        "maintenance_api", "internal_api_v1",
    ])
    def test_admin_name_blocked(self, balanced, name):
        r = validate_tool_call(tc(name, {}), balanced)
        assert r.allowed is False
        assert r.score == 0.80

    def test_admin_name_category(self, balanced):
        r = validate_tool_call(tc("admin_panel", {}), balanced)
        assert "dangerous_operation" in r.metadata.get("categories", [])

    def test_admin_name_blocked_by_logging_only(self, logging_only):
        # score=0.80 < block=1.0 for LoggingOnly → allowed
        r = validate_tool_call(tc("admin_panel", {}), logging_only)
        assert r.allowed is True

    def test_regular_name_not_flagged(self, balanced):
        r = validate_tool_call(tc("search_web", {"query": "test"}), balanced)
        assert r.allowed is True
        assert r.score == 0.0

class TestValidateToolCallPathTraversal:
    def test_dotdot_slash_blocked(self, balanced):
        r = validate_tool_call(tc("read_file", {"path": "../../etc/passwd"}), balanced)
        assert r.allowed is False
        assert r.score == 0.92

    def test_three_levels_up(self, balanced):
        r = validate_tool_call(tc("read_file", {"path": "../../../secret.txt"}), balanced)
        assert r.allowed is False
        assert r.score == 0.92

    def test_path_traversal_category(self, balanced):
        r = validate_tool_call(tc("read_file", {"path": "../../etc/shadow"}), balanced)
        assert "path_traversal" in r.metadata.get("categories", [])

    def test_safe_relative_path_not_flagged(self, balanced):
        r = validate_tool_call(tc("read_file", {"path": "documents/report.pdf"}), balanced)
        assert r.allowed is True

    def test_path_traversal_reason_mentions_key(self, balanced):
        r = validate_tool_call(tc("read_file", {"path": "../../etc/passwd"}), balanced)
        assert any("path" in reason for reason in r.reasons)

class TestValidateToolCallSystemPaths:
    @pytest.mark.parametrize("path", [
        "/etc/passwd", "/etc/shadow", "/proc/self/environ",
        "/sys/kernel/debug", "/dev/sda",
    ])
    def test_system_path_blocked(self, balanced, path):
        r = validate_tool_call(tc("read_file", {"path": path}), balanced)
        assert r.allowed is False
        assert r.score == 0.88

    def test_system_path_category(self, balanced):
        r = validate_tool_call(tc("read_file", {"path": "/etc/passwd"}), balanced)
        assert "dangerous_operation" in r.metadata.get("categories", [])

class TestValidateToolCallSensitiveFiles:
    @pytest.mark.parametrize("path", [
        "/home/user/.ssh/id_rsa",
        "/home/user/.env",
        "/home/user/.aws/credentials",
        "/var/www/.bash_history",
    ])
    def test_sensitive_file_blocked(self, balanced, path):
        r = validate_tool_call(tc("read_file", {"path": path}), balanced)
        assert r.allowed is False
        assert r.score == 0.95

    def test_sensitive_file_category(self, balanced):
        r = validate_tool_call(tc("read_file", {"path": "/home/user/.ssh/id_rsa"}), balanced)
        assert "dangerous_operation" in r.metadata.get("categories", [])

class TestValidateToolCallSSRF:
    def test_loopback_ip_blocked(self, balanced):
        r = validate_tool_call(tc("http_request", {"url": "http://127.0.0.1/admin"}), balanced)
        assert r.allowed is False
        assert r.score == 0.90

    def test_private_192_ip_blocked(self, balanced):
        r = validate_tool_call(tc("http_request", {"url": "http://192.168.1.1/api"}), balanced)
        assert r.allowed is False
        assert r.score == 0.90

    def test_private_10_ip_blocked(self, balanced):
        r = validate_tool_call(tc("http_request", {"url": "http://10.0.0.1/internal"}), balanced)
        assert r.allowed is False
        assert r.score == 0.90

    def test_aws_imds_blocked(self, balanced):
        r = validate_tool_call(
            tc("http_request", {"url": "http://169.254.169.254/latest/meta-data/"}),
            balanced,
        )
        assert r.allowed is False

    def test_gcp_metadata_blocked(self, balanced):
        r = validate_tool_call(
            tc("http_request", {"url": "http://metadata.google.internal/computeMetadata/v1/"}),
            balanced,
        )
        assert r.allowed is False
        assert r.score == 0.98

    def test_localhost_blocked(self, balanced):
        r = validate_tool_call(
            tc("http_request", {"url": "http://localhost:8080/admin"}),
            balanced,
        )
        assert r.allowed is False
        assert r.score == 0.88

    def test_ssrf_category_in_metadata(self, balanced):
        r = validate_tool_call(tc("http_request", {"url": "http://127.0.0.1/"}), balanced)
        assert "ssrf_attempt" in r.metadata.get("categories", [])

    def test_public_url_not_flagged(self, balanced):
        r = validate_tool_call(
            tc("http_request", {"url": "https://api.example.com/data"}),
            balanced,
        )
        assert r.allowed is True

class TestValidateToolCallCommandInj:
    def test_semicolon_injection(self, balanced):
        r = validate_tool_call(tc("search_web", {"query": "python; rm -rf /"}), balanced)
        assert r.allowed is False
        assert r.score == 0.88

    def test_double_ampersand_injection(self, balanced):
        r = validate_tool_call(
            tc("search_web", {"query": "test && malicious_cmd"}), balanced
        )
        assert r.allowed is False
        assert r.score == 0.88

    def test_pipe_injection(self, balanced):
        r = validate_tool_call(
            tc("search_web", {"query": "test | bash -x /tmp/evil.sh"}), balanced
        )
        assert r.allowed is False
        assert r.score == 0.88

    def test_subshell_dollar_paren(self, balanced):
        r = validate_tool_call(
            tc("search_web", {"query": "$(cat /etc/passwd)"}), balanced
        )
        assert r.allowed is False
        assert r.score == 0.88

    def test_command_injection_category(self, balanced):
        r = validate_tool_call(tc("search_web", {"query": "test; id"}), balanced)
        assert "command_injection" in r.metadata.get("categories", [])

    def test_clean_query_not_flagged(self, balanced):
        r = validate_tool_call(tc("search_web", {"query": "python best practices"}), balanced)
        assert r.allowed is True

class TestValidateToolCallSqlInj:
    def test_or_1_equals_1(self, balanced):
        r = validate_tool_call(
            tc("db_query", {"query": "SELECT * FROM users WHERE id=1 OR 1=1"}),
            balanced,
        )
        assert r.allowed is False
        assert r.score == 0.90

    def test_union_select(self, balanced):
        r = validate_tool_call(
            tc("db_query", {"query": "SELECT * FROM users UNION SELECT * FROM admins"}),
            balanced,
        )
        assert r.allowed is False
        assert r.score == 0.90

    def test_drop_table(self, balanced):
        r = validate_tool_call(
            tc("db_query", {"query": "SELECT 1; DROP TABLE users"}),
            balanced,
        )
        assert r.allowed is False

    def test_safe_sql_not_flagged(self, balanced):
        r = validate_tool_call(
            tc("db_query", {"query": "SELECT name, age FROM users WHERE id = 42"}),
            balanced,
        )
        assert r.allowed is True

class TestValidateToolCallSchemaValid:
    def test_valid_args_allowed(self, balanced):
        r = validate_tool_call(
            tc("create_user", {"username": "alice", "age": 30, "limit": 50},
               schema=_FULL_SCHEMA),
            balanced,
        )
        assert r.allowed is True
        assert r.score == 0.0

    def test_schema_validated_flag_true(self, balanced):
        r = validate_tool_call(
            tc("create_user", {"username": "alice", "age": 30}, schema=_FULL_SCHEMA),
            balanced,
        )
        assert r.metadata.get("schema_validated") is True

    def test_no_schema_flag_false(self, balanced):
        r = validate_tool_call(tc("search_web", {"query": "test"}), balanced)
        assert r.metadata.get("schema_validated") is False

    def test_extra_fields_not_penalised(self, balanced):
        # JSON Schema allows additional properties by default
        r = validate_tool_call(
            tc("create_user",
               {"username": "alice", "age": 30, "extra_field": "ok"},
               schema=_FULL_SCHEMA),
            balanced,
        )
        assert r.allowed is True

class TestValidateToolCallSchemaViolation:
    def test_type_mismatch_blocked(self, balanced):
        r = validate_tool_call(
            tc("create_user", {"username": "alice", "age": "thirty"}, schema=_FULL_SCHEMA),
            balanced,
        )
        assert r.allowed is False
        assert r.score == 0.88

    def test_type_mismatch_category(self, balanced):
        r = validate_tool_call(
            tc("create_user", {"username": "alice", "age": "thirty"}, schema=_FULL_SCHEMA),
            balanced,
        )
        assert "schema_violation" in r.metadata.get("categories", [])

    def test_missing_required_field_blocked(self, balanced):
        r = validate_tool_call(
            tc("create_user", {"username": "alice"}, schema=_FULL_SCHEMA),
            balanced,
        )
        assert r.allowed is False
        assert r.score == 0.88

    def test_minimum_violation_blocked(self, balanced):
        r = validate_tool_call(
            tc("create_user", {"username": "alice", "age": 25, "limit": 0},
               schema=_FULL_SCHEMA),
            balanced,
        )
        assert r.allowed is False
        assert r.score == 0.88

    def test_maximum_violation_blocked(self, balanced):
        r = validate_tool_call(
            tc("create_user", {"username": "alice", "age": 25, "limit": 200},
               schema=_FULL_SCHEMA),
            balanced,
        )
        assert r.allowed is False
        assert r.score == 0.88

    def test_enum_violation_blocked(self, balanced):
        enum_schema = {
            "type": "object",
            "properties": {"color": {"type": "string", "enum": ["red", "green", "blue"]}},
        }
        r = validate_tool_call(
            tc("set_color", {"color": "purple"}, schema=enum_schema),
            balanced,
        )
        assert r.allowed is False
        assert r.score == 0.88

    def test_schema_violation_reason_is_string(self, balanced):
        r = validate_tool_call(
            tc("create_user", {"username": "alice", "age": "thirty"}, schema=_FULL_SCHEMA),
            balanced,
        )
        assert all(isinstance(reason, str) for reason in r.reasons)

class TestValidateToolCallResultStructure:
    def test_guard_type_is_tool(self, balanced):
        r = validate_tool_call(tc("search_web", {"query": "hello"}), balanced)
        assert r.guard_type == GuardType.TOOL

    def test_metadata_has_tool_name(self, balanced):
        r = validate_tool_call(tc("read_file", {"path": "../../etc/passwd"}), balanced)
        assert r.metadata.get("tool_name") == "read_file"

    def test_metadata_has_call_id(self, balanced):
        r = validate_tool_call(
            tc("read_file", {"path": "../../etc/passwd"}, call_id="call_abc"),
            balanced,
        )
        assert r.metadata.get("call_id") == "call_abc"

    def test_metadata_call_id_none_when_not_set(self, balanced):
        r = validate_tool_call(tc("search_web", {"query": "test"}), balanced)
        assert r.metadata.get("call_id") is None

    def test_metadata_has_categories(self, balanced):
        r = validate_tool_call(tc("read_file", {"path": "../../etc/passwd"}), balanced)
        assert "categories" in r.metadata

    def test_metadata_has_check_count(self, balanced):
        r = validate_tool_call(tc("search_web", {"query": "test"}), balanced)
        assert "check_count" in r.metadata
        assert isinstance(r.metadata["check_count"], int)

    def test_metadata_has_schema_validated(self, balanced):
        r = validate_tool_call(tc("search_web", {"query": "test"}), balanced)
        assert "schema_validated" in r.metadata

    def test_reasons_is_list(self, balanced):
        r = validate_tool_call(tc("read_file", {"path": "../../etc/passwd"}), balanced)
        assert isinstance(r.reasons, list)

    def test_score_is_float(self, balanced):
        r = validate_tool_call(tc("read_file", {"path": "../../etc/passwd"}), balanced)
        assert isinstance(r.score, float)

    def test_path_traversal_category_in_blocked_result(self, balanced):
        r = validate_tool_call(tc("read_file", {"path": "../../etc/passwd"}), balanced)
        assert "path_traversal" in r.metadata.get("categories", [])

    def test_safe_result_check_count_at_least_3(self, balanced):
        # At minimum the 3 name-level checks run for a safe call
        r = validate_tool_call(tc("search_web", {"query": "python"}), balanced)
        assert r.metadata["check_count"] >= 3

    def test_schema_validated_true_when_schema_provided(self, balanced):
        schema = {"type": "object", "properties": {"q": {"type": "string"}}}
        r = validate_tool_call(tc("search_web", {"q": "test"}, schema=schema), balanced)
        assert r.metadata.get("schema_validated") is True

class TestValidateToolCallGuardDisabled:
    def test_disabled_allows_dangerous_name(self, disabled_policy):
        r = validate_tool_call(tc("exec", {}), disabled_policy)
        assert r.allowed is True
        assert r.score == 0.0

    def test_disabled_allows_path_traversal(self, disabled_policy):
        r = validate_tool_call(
            tc("read_file", {"path": "../../etc/passwd"}), disabled_policy
        )
        assert r.allowed is True

    def test_disabled_sets_skipped_metadata(self, disabled_policy):
        r = validate_tool_call(tc("exec", {}), disabled_policy)
        assert r.metadata.get("skipped") is True

    def test_disabled_guard_type_still_tool(self, disabled_policy):
        r = validate_tool_call(tc("exec", {}), disabled_policy)
        assert r.guard_type == GuardType.TOOL

    def test_disabled_reasons_empty(self, disabled_policy):
        r = validate_tool_call(tc("exec", {}), disabled_policy)
        assert r.reasons == []

    def test_disabled_tool_name_in_metadata(self, disabled_policy):
        r = validate_tool_call(tc("exec", {}), disabled_policy)
        assert r.metadata.get("tool_name") == "exec"

class TestValidateToolCallShortCircuit:
    def test_short_circuit_true_stops_early(self, balanced):
        r = validate_tool_call(
            tc("read_file", {"path": "../../etc/passwd"}),
            balanced, short_circuit=True,
        )
        r_all = validate_tool_call(
            tc("read_file", {"path": "../../etc/passwd"}),
            balanced, short_circuit=False,
        )
        assert r.metadata["check_count"] < r_all.metadata["check_count"]

    def test_short_circuit_true_still_blocks(self, balanced):
        r = validate_tool_call(
            tc("read_file", {"path": "../../etc/passwd"}),
            balanced, short_circuit=True,
        )
        assert r.allowed is False

    def test_short_circuit_false_still_blocks(self, balanced):
        r = validate_tool_call(
            tc("read_file", {"path": "../../etc/passwd"}),
            balanced, short_circuit=False,
        )
        assert r.allowed is False

    def test_short_circuit_false_collects_more_violations(self, balanced):
        # A call that triggers multiple checks — short_circuit=False finds all
        text = "../../etc/passwd; DROP TABLE users"
        r_sc  = validate_tool_call(tc("read_file", {"path": text}), balanced, short_circuit=True)
        r_all = validate_tool_call(tc("read_file", {"path": text}), balanced, short_circuit=False)
        assert len(r_all.reasons) >= len(r_sc.reasons)

class TestValidateToolCallNestedArgs:
    def test_path_traversal_in_nested_dict(self, balanced):
        r = validate_tool_call(
            tc("process", {"config": {"path": "../../etc/passwd"}}),
            balanced,
        )
        assert r.allowed is False
        assert r.score == 0.92

    def test_ssrf_in_nested_dict(self, balanced):
        r = validate_tool_call(
            tc("fetch", {"options": {"url": "http://127.0.0.1/admin"}}),
            balanced,
        )
        assert r.allowed is False

    def test_safe_nested_dict_passes(self, balanced):
        r = validate_tool_call(
            tc("process", {"config": {"path": "documents/report.pdf", "format": "pdf"}}),
            balanced,
        )
        assert r.allowed is True

    def test_injection_in_list_item(self, balanced):
        r = validate_tool_call(
            tc("batch_read", {"paths": ["../../etc/passwd", "safe.txt"]}),
            balanced,
        )
        assert r.allowed is False

class TestValidateToolCallPolicyThresholds:
    def test_balanced_blocks_path_traversal(self, balanced):
        r = validate_tool_call(tc("read_file", {"path": "../../etc/passwd"}), balanced)
        assert r.allowed is False

    def test_strict_blocks_admin_name(self, strict):
        # admin_panel score=0.80 >= strict block=0.40
        r = validate_tool_call(tc("admin_panel", {}), strict)
        assert r.allowed is False

    def test_balanced_blocks_admin_name(self, balanced):
        # admin_panel score=0.80 >= balanced block=0.75
        r = validate_tool_call(tc("admin_panel", {}), balanced)
        assert r.allowed is False

    def test_logging_only_allows_admin_name(self, logging_only):
        # admin_panel score=0.80 < logging block=1.0
        r = validate_tool_call(tc("admin_panel", {}), logging_only)
        assert r.allowed is True

    def test_logging_only_still_computes_score(self, logging_only):
        r = validate_tool_call(tc("admin_panel", {}), logging_only)
        assert r.score == 0.80

    def test_balanced_blocks_dangerous_name(self, balanced):
        r = validate_tool_call(tc("exec", {}), balanced)
        assert r.allowed is False

    def test_logging_only_allows_dangerous_name(self, logging_only):
        r = validate_tool_call(tc("exec", {}), logging_only)
        assert r.allowed is True

    def test_per_guard_threshold_override(self):
        p = BalancedPolicy(
            tool_guard=GuardConfig(block_threshold=0.98),
            raise_on_block=False,
        )
        # exec score=0.95 < per-guard override 0.98 → allowed
        r = validate_tool_call(tc("exec", {}), p)
        assert r.allowed is True
        assert r.score == 0.95

class TestValidateToolCalls:
    def test_returns_one_result_per_call(self, balanced):
        calls = [
            tc("search_web", {"query": "python"}),
            tc("read_file",  {"path": "../../etc/passwd"}),
        ]
        results = validate_tool_calls(calls, balanced)
        assert len(results) == 2

    def test_safe_call_result_allowed(self, balanced):
        results = validate_tool_calls(
            [tc("search_web", {"query": "python"})], balanced
        )
        assert results[0].allowed is True

    def test_dangerous_call_result_blocked(self, balanced):
        results = validate_tool_calls(
            [tc("read_file", {"path": "../../etc/passwd"})], balanced
        )
        assert results[0].allowed is False

    def test_empty_list_returns_empty(self, balanced):
        assert validate_tool_calls([], balanced) == []

    def test_order_preserved(self, balanced):
        calls = [
            tc("search_web", {"query": "python"}),
            tc("exec", {}),
            tc("get_weather", {"city": "Paris"}),
        ]
        results = validate_tool_calls(calls, balanced)
        assert results[0].metadata["tool_name"] == "search_web"
        assert results[1].metadata["tool_name"] == "exec"
        assert results[2].metadata["tool_name"] == "get_weather"

    def test_mix_of_safe_and_dangerous(self, balanced):
        calls = [
            tc("search_web", {"query": "hello"}),
            tc("read_file",  {"path": "../../etc/passwd"}),
        ]
        results = validate_tool_calls(calls, balanced)
        assert results[0].allowed is True
        assert results[1].allowed is False

    def test_all_results_are_scan_results(self, balanced):
        calls = [tc("search_web", {"query": "a"}), tc("exec", {})]
        for r in validate_tool_calls(calls, balanced):
            assert isinstance(r, ScanResult)

class TestAggregateToolScans:
    def test_empty_list_returns_clean(self, balanced):
        agg = aggregate_tool_scans([], balanced)
        assert agg.allowed is True
        assert agg.score == 0.0

    def test_empty_list_source_count_zero(self, balanced):
        agg = aggregate_tool_scans([], balanced)
        assert agg.metadata.get("source_count") == 0

    def test_all_safe_results_aggregate_clean(self, balanced):
        r1 = validate_tool_call(tc("search_web", {"query": "python"}), balanced)
        r2 = validate_tool_call(tc("get_weather", {"city": "Paris"}), balanced)
        agg = aggregate_tool_scans([r1, r2], balanced)
        assert agg.allowed is True
        assert agg.score == 0.0

    def test_one_blocked_makes_aggregate_blocked(self, balanced):
        r_safe = validate_tool_call(tc("search_web", {"query": "python"}), balanced)
        r_bad  = validate_tool_call(tc("read_file", {"path": "../../etc/passwd"}), balanced)
        agg = aggregate_tool_scans([r_safe, r_bad], balanced)
        assert agg.allowed is False

    def test_aggregate_score_is_max(self, balanced):
        r_high = validate_tool_call(tc("exec", {}), balanced)          # 0.95
        r_low  = validate_tool_call(tc("admin_panel", {}), balanced)   # 0.80
        agg = aggregate_tool_scans([r_high, r_low], balanced)
        assert agg.score == max(r_high.score, r_low.score)
        assert agg.score == 0.95

    def test_reasons_merged(self, balanced):
        r1 = validate_tool_call(tc("exec", {}), balanced)
        r2 = validate_tool_call(tc("read_file", {"path": "../../etc/passwd"}), balanced)
        agg = aggregate_tool_scans([r1, r2], balanced)
        assert len(agg.reasons) >= 2

    def test_guard_type_is_tool(self, balanced):
        r = validate_tool_call(tc("search_web", {"query": "test"}), balanced)
        agg = aggregate_tool_scans([r], balanced)
        assert agg.guard_type == GuardType.TOOL

    def test_metadata_aggregated_flag(self, balanced):
        r = validate_tool_call(tc("search_web", {"query": "test"}), balanced)
        agg = aggregate_tool_scans([r], balanced)
        assert agg.metadata.get("aggregated") is True

    def test_metadata_source_count(self, balanced):
        r1 = validate_tool_call(tc("search_web", {"query": "a"}), balanced)
        r2 = validate_tool_call(tc("search_web", {"query": "b"}), balanced)
        agg = aggregate_tool_scans([r1, r2], balanced)
        assert agg.metadata.get("source_count") == 2

    def test_metadata_tool_names_list(self, balanced):
        r_safe = validate_tool_call(tc("search_web", {"query": "python"}), balanced)
        r_bad  = validate_tool_call(tc("read_file", {"path": "../../etc/passwd"}), balanced)
        agg = aggregate_tool_scans([r_safe, r_bad], balanced)
        tool_names = agg.metadata.get("tool_names", [])
        assert "search_web" in tool_names
        assert "read_file" in tool_names

    def test_single_result_aggregate(self, balanced):
        r = validate_tool_call(tc("exec", {}), balanced)
        agg = aggregate_tool_scans([r], balanced)
        assert agg.allowed is False
        assert agg.score == r.score

    def test_aggregate_respects_policy_threshold(self, logging_only):
        # exec score=0.95 < logging block=1.0 → allowed even in aggregate
        r = validate_tool_call(tc("exec", {}), logging_only)
        agg = aggregate_tool_scans([r], logging_only)
        assert agg.allowed is True

    def test_categories_deduped(self, balanced):
        r1 = validate_tool_call(tc("read_file", {"path": "../../etc/passwd"}), balanced)
        r2 = validate_tool_call(tc("read_file", {"path": "../secret.txt"}), balanced)
        agg = aggregate_tool_scans([r1, r2], balanced)
        cats = agg.metadata.get("categories", [])
        assert len(cats) == len(set(cats))

    def test_from_validate_tool_calls_output(self, balanced):
        calls = [
            tc("search_web", {"query": "python"}),
            tc("read_file",  {"path": "../../etc/passwd"}),
        ]
        per_call = validate_tool_calls(calls, balanced)
        agg = aggregate_tool_scans(per_call, balanced)
        assert agg.allowed is False
        assert agg.score == 0.92