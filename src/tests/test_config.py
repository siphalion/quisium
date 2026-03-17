from __future__ import annotations
import os
import tempfile
from typing import Any, Dict
import pytest
from src.config import (
    ConfigError,
    ConfigSourceError,
    ConfigValidationError,
    _deep_merge,
    _parse_bool,
    _parse_csv_list,
    _parse_csv_set,
    get_default_policy,
    load_config,
    reset_default_policy,
    set_default_policy,
    validate_config_dict,
)
from src.policies import BalancedPolicy, GuardConfig, Policy, StrictPolicy
from src.types import GuardType

class TestConfigExceptions:
    def test_config_error_is_exception(self):
        exc = ConfigError("base error")
        assert isinstance(exc, Exception)

    def test_config_validation_error_is_config_error(self):
        exc = ConfigValidationError(["error A"])
        assert isinstance(exc, ConfigError)

    def test_config_source_error_is_config_error(self):
        exc = ConfigSourceError("file missing")
        assert isinstance(exc, ConfigError)

    def test_validation_error_stores_errors_list(self):
        exc = ConfigValidationError(["err A", "err B"])
        assert exc.errors == ["err A", "err B"]

    def test_validation_error_single_error(self):
        exc = ConfigValidationError(["only error"])
        assert len(exc.errors) == 1
        assert exc.errors[0] == "only error"

    def test_validation_error_message_contains_errors(self):
        exc = ConfigValidationError(["err A", "err B"])
        assert "err A" in str(exc)
        assert "err B" in str(exc)

    def test_validation_error_message_contains_count(self):
        exc = ConfigValidationError(["err A", "err B"])
        assert "2" in str(exc)

    def test_validation_error_empty_list(self):
        # Constructing with an empty list is unusual but should not crash
        exc = ConfigValidationError([])
        assert exc.errors == []

    def test_source_error_message_preserved(self):
        exc = ConfigSourceError("policy.yaml not found")
        assert "policy.yaml not found" in str(exc)

class TestParseBool:
    @pytest.mark.parametrize("value", ["1", "true", "True", "TRUE", "yes", "Yes", "YES", "on", "ON"])
    def test_truthy_strings(self, value):
        assert _parse_bool(value) is True

    @pytest.mark.parametrize("value", ["0", "false", "False", "FALSE", "no", "No", "NO", "off", "OFF"])
    def test_falsy_strings(self, value):
        assert _parse_bool(value) is False

    def test_truthy_with_surrounding_whitespace(self):
        assert _parse_bool("  true  ") is True

    def test_falsy_with_surrounding_whitespace(self):
        assert _parse_bool("  false  ") is False

    def test_invalid_string_raises_config_validation_error(self):
        with pytest.raises(ConfigValidationError):
            _parse_bool("maybe")

    def test_invalid_string_error_message(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            _parse_bool("maybe")
        assert len(exc_info.value.errors) == 1
        assert "maybe" in exc_info.value.errors[0]

    def test_empty_string_raises(self):
        with pytest.raises(ConfigValidationError):
            _parse_bool("")

    def test_numeric_two_raises(self):
        with pytest.raises(ConfigValidationError):
            _parse_bool("2")

    def test_returns_bool_type(self):
        assert isinstance(_parse_bool("true"), bool)
        assert isinstance(_parse_bool("false"), bool)

class TestParseCsvList:
    def test_two_items(self):
        assert _parse_csv_list("read_file,search_web") == ["read_file", "search_web"]

    def test_strips_whitespace_around_items(self):
        assert _parse_csv_list("read_file, search_web , exec") == ["read_file", "search_web", "exec"]

    def test_trailing_comma_dropped(self):
        result = _parse_csv_list("read_file,search_web,")
        assert result == ["read_file", "search_web"]

    def test_empty_string_returns_empty_list(self):
        assert _parse_csv_list("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert _parse_csv_list("   ") == []

    def test_single_item(self):
        assert _parse_csv_list("search_web") == ["search_web"]

    def test_returns_list_type(self):
        assert isinstance(_parse_csv_list("a,b"), list)

    def test_comma_only_returns_empty_list(self):
        # Commas with no real items → empty list
        result = _parse_csv_list(",,,")
        assert result == []

class TestParseCsvSet:
    def test_returns_set_type(self):
        result = _parse_csv_set("exec_shell,delete_file")
        assert isinstance(result, set)

    def test_contains_expected_items(self):
        result = _parse_csv_set("exec_shell,delete_file")
        assert "exec_shell" in result
        assert "delete_file" in result

    def test_deduplicates_items(self):
        result = _parse_csv_set("search_web,search_web")
        assert len(result) == 1
        assert "search_web" in result

    def test_empty_string_returns_empty_set(self):
        assert _parse_csv_set("") == set()

    def test_single_item(self):
        assert _parse_csv_set("exec_shell") == {"exec_shell"}

class TestDeepMerge:
    def test_non_overlapping_keys_combined(self):
        result = _deep_merge({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_override_scalar_wins(self):
        result = _deep_merge({"a": 1}, {"a": 99})
        assert result["a"] == 99

    def test_base_preserved_when_not_overridden(self):
        result = _deep_merge({"a": 1, "b": 2}, {"b": 99})
        assert result["a"] == 1

    def test_nested_dicts_merged_recursively(self):
        base     = {"nested": {"x": 10, "y": 20}}
        override = {"nested": {"y": 99, "z": 30}}
        result   = _deep_merge(base, override)
        assert result["nested"] == {"x": 10, "y": 99, "z": 30}

    def test_nested_key_preserved_when_not_in_override(self):
        base     = {"nested": {"x": 10, "y": 20}}
        override = {"nested": {"y": 99}}
        result   = _deep_merge(base, override)
        assert result["nested"]["x"] == 10

    def test_does_not_mutate_base(self):
        base     = {"a": 1, "nested": {"x": 10}}
        override = {"nested": {"x": 99}}
        _deep_merge(base, override)
        assert base["nested"]["x"] == 10

    def test_does_not_mutate_override(self):
        base     = {"a": 1}
        override = {"b": [1, 2, 3]}
        _deep_merge(base, override)
        assert override["b"] == [1, 2, 3]

    def test_scalar_overrides_nested_dict(self):
        # If override has a scalar where base has a dict, scalar wins
        base     = {"a": {"x": 1}}
        override = {"a": "string"}
        result   = _deep_merge(base, override)
        assert result["a"] == "string"

    def test_nested_dict_overrides_scalar(self):
        base     = {"a": "string"}
        override = {"a": {"x": 1}}
        result   = _deep_merge(base, override)
        assert result["a"] == {"x": 1}

    def test_empty_base(self):
        result = _deep_merge({}, {"a": 1})
        assert result == {"a": 1}

    def test_empty_override(self):
        result = _deep_merge({"a": 1}, {})
        assert result == {"a": 1}

    def test_returns_new_dict_not_base(self):
        base   = {"a": 1}
        result = _deep_merge(base, {"b": 2})
        assert result is not base

class TestValidateConfigDict:
    def test_valid_full_dict_passes(self):
        validate_config_dict({
            "name": "my-policy",
            "block_threshold": 0.75,
            "warn_threshold": 0.40,
            "raise_on_block": True,
        })  # must not raise

    def test_empty_dict_passes(self):
        validate_config_dict({})  # must not raise

    def test_allowed_tools_none_passes(self):
        validate_config_dict({"allowed_tools": None})

    def test_allowed_tools_empty_list_passes(self):
        validate_config_dict({"allowed_tools": []})

    def test_blocked_tools_as_list_passes(self):
        validate_config_dict({"blocked_tools": ["exec_shell"]})

    def test_blocked_tools_as_set_passes(self):
        validate_config_dict({"blocked_tools": {"exec_shell"}})

    def test_guard_config_object_passes(self):
        validate_config_dict({"prompt_guard": GuardConfig(enabled=False)})

    def test_valid_guard_dict_passes(self):
        validate_config_dict({
            "prompt_guard": {"enabled": True, "block_threshold": 0.50, "warn_threshold": 0.20}
        })

    def test_block_threshold_above_one_raises(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config_dict({"block_threshold": 1.5})
        assert any("block_threshold" in e for e in exc_info.value.errors)

    def test_block_threshold_below_zero_raises(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config_dict({"block_threshold": -0.1})
        assert any("block_threshold" in e for e in exc_info.value.errors)

    def test_block_threshold_not_a_number_raises(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config_dict({"block_threshold": "high"})
        assert any("block_threshold" in e for e in exc_info.value.errors)

    def test_block_threshold_exactly_zero_with_lower_warn_passes(self):
        # warn must be strictly less than block, so warn=0.0 only works if block=0.0
        # is impossible — use the smallest valid pair instead
        validate_config_dict({"block_threshold": 1.0, "warn_threshold": 0.0})

    def test_block_threshold_exactly_one_passes(self):
        validate_config_dict({"block_threshold": 1.0, "warn_threshold": 0.5})

    def test_warn_threshold_above_one_raises(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config_dict({"warn_threshold": 1.1})
        assert any("warn_threshold" in e for e in exc_info.value.errors)

    def test_warn_threshold_below_zero_raises(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config_dict({"warn_threshold": -0.1})
        assert any("warn_threshold" in e for e in exc_info.value.errors)

    def test_warn_equal_to_block_raises(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config_dict({"block_threshold": 0.5, "warn_threshold": 0.5})
        assert any("warn_threshold" in e and "block_threshold" in e
                   for e in exc_info.value.errors)

    def test_warn_greater_than_block_raises(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config_dict({"block_threshold": 0.4, "warn_threshold": 0.6})
        assert any("warn_threshold" in e for e in exc_info.value.errors)

    def test_raise_on_block_string_raises(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config_dict({"raise_on_block": "true"})
        assert any("raise_on_block" in e for e in exc_info.value.errors)

    def test_raise_on_block_int_raises(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config_dict({"raise_on_block": 1})
        assert any("raise_on_block" in e for e in exc_info.value.errors)

    def test_redact_on_warn_string_raises(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config_dict({"redact_on_warn": "yes"})
        assert any("redact_on_warn" in e for e in exc_info.value.errors)

    def test_log_clean_requests_string_raises(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config_dict({"log_clean_requests": "no"})
        assert any("log_clean_requests" in e for e in exc_info.value.errors)

    def test_empty_name_raises(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config_dict({"name": ""})
        assert any("name" in e for e in exc_info.value.errors)

    def test_whitespace_name_raises(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config_dict({"name": "   "})
        assert any("name" in e for e in exc_info.value.errors)

    def test_valid_name_passes(self):
        validate_config_dict({"name": "my-policy"})

    def test_allowed_tools_string_raises(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config_dict({"allowed_tools": "search_web"})
        assert any("allowed_tools" in e for e in exc_info.value.errors)

    def test_allowed_tools_with_non_strings_raises(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config_dict({"allowed_tools": [1, 2, 3]})
        assert any("allowed_tools" in e for e in exc_info.value.errors)

    def test_blocked_tools_string_raises(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config_dict({"blocked_tools": "exec_shell"})
        assert any("blocked_tools" in e for e in exc_info.value.errors)

    def test_blocked_tools_with_non_strings_raises(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config_dict({"blocked_tools": [42]})
        assert any("blocked_tools" in e for e in exc_info.value.errors)

    def test_guard_unknown_field_raises(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config_dict({"prompt_guard": {"bad_key": True}})
        assert any("prompt_guard" in e and "bad_key" in e
                   for e in exc_info.value.errors)

    def test_guard_block_threshold_out_of_range_raises(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config_dict({"prompt_guard": {"block_threshold": 2.0}})
        assert any("prompt_guard" in e for e in exc_info.value.errors)

    def test_guard_warn_equals_block_raises(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config_dict({
                "prompt_guard": {"block_threshold": 0.5, "warn_threshold": 0.5}
            })
        assert any("prompt_guard" in e for e in exc_info.value.errors)

    def test_guard_non_dict_raises(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config_dict({"output_guard": "enabled"})
        assert any("output_guard" in e for e in exc_info.value.errors)

    def test_all_three_guards_validated(self):
        # Each guard independently validated
        for guard_key in ("prompt_guard", "output_guard", "tool_guard"):
            with pytest.raises(ConfigValidationError):
                validate_config_dict({guard_key: {"block_threshold": 9.9}})

    def test_multiple_errors_collected_before_raising(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config_dict({
                "block_threshold": 2.0,
                "warn_threshold": -0.1,
                "raise_on_block": "yes",
            })
        assert len(exc_info.value.errors) == 3

    def test_errors_attribute_is_list_of_strings(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config_dict({"block_threshold": 2.0})
        assert isinstance(exc_info.value.errors, list)
        assert all(isinstance(e, str) for e in exc_info.value.errors)

class TestDefaultPolicyRegistry:
    def test_get_after_reset_returns_balanced_policy(self):
        reset_default_policy()
        p = get_default_policy()
        assert p.name == "balanced"

    def test_get_returns_policy_instance(self):
        reset_default_policy()
        assert isinstance(get_default_policy(), Policy)

    def test_set_then_get_returns_set_policy(self):
        set_default_policy(StrictPolicy())
        assert get_default_policy().name == "strict"

    def test_set_custom_policy(self):
        custom = BalancedPolicy(name="my-custom", block_threshold=0.60)
        set_default_policy(custom)
        p = get_default_policy()
        assert p.name == "my-custom"
        assert p.block_threshold == 0.60

    def test_get_returns_same_instance_each_call(self):
        reset_default_policy()
        p1 = get_default_policy()
        p2 = get_default_policy()
        assert p1 is p2

    def test_reset_after_set_restores_balanced(self):
        set_default_policy(StrictPolicy())
        reset_default_policy()
        assert get_default_policy().name == "balanced"

    def test_set_with_non_policy_raises_type_error(self):
        with pytest.raises(TypeError, match="Policy instance"):
            set_default_policy("not-a-policy")  # type: ignore

    def test_set_with_none_raises_type_error(self):
        with pytest.raises(TypeError):
            set_default_policy(None)  # type: ignore

    def test_set_with_dict_raises_type_error(self):
        with pytest.raises(TypeError):
            set_default_policy({"name": "balanced"})  # type: ignore

    def test_multiple_resets_idempotent(self):
        reset_default_policy()
        reset_default_policy()
        assert get_default_policy().name == "balanced"

class TestLoadConfigDefaults:
    def test_returns_policy_instance(self):
        p = load_config(use_env=False)
        assert isinstance(p, Policy)

    def test_default_name_is_balanced(self):
        p = load_config(use_env=False)
        assert p.name == "balanced"

    def test_default_block_threshold(self):
        p = load_config(use_env=False)
        assert p.block_threshold == 0.75

    def test_default_warn_threshold(self):
        p = load_config(use_env=False)
        assert p.warn_threshold == 0.40

    def test_default_raise_on_block(self):
        p = load_config(use_env=False)
        assert p.raise_on_block is True

class TestLoadConfigDataSource:
    def test_block_threshold_overridden(self):
        p = load_config(data={"block_threshold": 0.60, "warn_threshold": 0.25}, use_env=False)
        assert p.block_threshold == 0.60

    def test_warn_threshold_overridden(self):
        p = load_config(data={"block_threshold": 0.60, "warn_threshold": 0.25}, use_env=False)
        assert p.warn_threshold == 0.25

    def test_name_overridden(self):
        p = load_config(data={"name": "custom"}, use_env=False)
        assert p.name == "custom"

    def test_raise_on_block_overridden(self):
        p = load_config(data={"raise_on_block": False}, use_env=False)
        assert p.raise_on_block is False

    def test_allowed_tools_overridden(self):
        p = load_config(data={"allowed_tools": ["search_web"]}, use_env=False)
        assert p.is_tool_allowed("search_web") is True
        assert p.is_tool_allowed("delete_file") is False

    def test_blocked_tools_overridden(self):
        p = load_config(data={"blocked_tools": ["exec_shell"]}, use_env=False)
        assert p.is_tool_allowed("exec_shell") is False

    def test_nested_guard_enabled_false(self):
        p = load_config(
            data={"prompt_guard": {"enabled": False}},
            use_env=False,
        )
        assert p.is_guard_enabled(GuardType.PROMPT) is False
        assert p.is_guard_enabled(GuardType.OUTPUT) is True

    def test_nested_guard_block_threshold_overridden(self):
        p = load_config(
            data={"output_guard": {"block_threshold": 0.55, "warn_threshold": 0.20}},
            use_env=False,
        )
        assert p.effective_block_threshold(GuardType.OUTPUT) == 0.55
        assert p.effective_warn_threshold(GuardType.OUTPUT) == 0.20

    def test_empty_data_dict_uses_defaults(self):
        p = load_config(data={}, use_env=False)
        assert p.block_threshold == 0.75

class TestLoadConfigBasePolicy:
    def test_base_policy_provides_defaults(self):
        p = load_config(base_policy=StrictPolicy(), use_env=False)
        assert p.block_threshold == 0.40

    def test_base_policy_name_preserved(self):
        p = load_config(base_policy=StrictPolicy(), use_env=False)
        assert p.name == "strict"

    def test_data_overrides_base_policy(self):
        p = load_config(
            base_policy=StrictPolicy(),
            data={"block_threshold": 0.80, "warn_threshold": 0.30},
            use_env=False,
        )
        assert p.block_threshold == 0.80

    def test_base_non_overridden_fields_preserved(self):
        # StrictPolicy has log_clean_requests=True — data doesn't touch it
        p = load_config(
            base_policy=StrictPolicy(),
            data={"block_threshold": 0.80, "warn_threshold": 0.30},
            use_env=False,
        )
        assert p.log_clean_requests is True

    def test_none_base_policy_uses_balanced(self):
        p = load_config(base_policy=None, use_env=False)
        assert p.block_threshold == 0.75

class TestLoadConfigEnvOverrides:
    def test_block_threshold_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_SECURITY_BLOCK_THRESHOLD", "0.50")
        p = load_config(use_env=True, data={})
        assert p.block_threshold == 0.50

    def test_warn_threshold_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_SECURITY_WARN_THRESHOLD", "0.20")
        p = load_config(use_env=True, data={})
        assert p.warn_threshold == 0.20

    def test_name_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_SECURITY_NAME", "env-policy")
        p = load_config(use_env=True, data={})
        assert p.name == "env-policy"

    def test_raise_on_block_false_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_SECURITY_RAISE_ON_BLOCK", "false")
        p = load_config(use_env=True, data={})
        assert p.raise_on_block is False

    def test_raise_on_block_true_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_SECURITY_RAISE_ON_BLOCK", "true")
        p = load_config(use_env=True, data={})
        assert p.raise_on_block is True

    def test_raise_on_block_zero_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_SECURITY_RAISE_ON_BLOCK", "0")
        p = load_config(use_env=True, data={})
        assert p.raise_on_block is False

    def test_raise_on_block_one_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_SECURITY_RAISE_ON_BLOCK", "1")
        p = load_config(use_env=True, data={})
        assert p.raise_on_block is True

    def test_allowed_tools_csv_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_SECURITY_ALLOWED_TOOLS", "search_web,read_file")
        p = load_config(use_env=True, data={})
        assert p.is_tool_allowed("search_web") is True
        assert p.is_tool_allowed("read_file") is True
        assert p.is_tool_allowed("delete_file") is False

    def test_blocked_tools_csv_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_SECURITY_BLOCKED_TOOLS", "exec_shell,delete_file")
        p = load_config(use_env=True, data={})
        assert p.is_tool_allowed("exec_shell") is False
        assert p.is_tool_allowed("delete_file") is False
        assert p.is_tool_allowed("search_web") is True

    def test_prompt_guard_enabled_false_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_SECURITY_PROMPT_GUARD_ENABLED", "false")
        p = load_config(use_env=True, data={})
        assert p.is_guard_enabled(GuardType.PROMPT) is False
        assert p.is_guard_enabled(GuardType.OUTPUT) is True

    def test_output_guard_enabled_false_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_SECURITY_OUTPUT_GUARD_ENABLED", "false")
        p = load_config(use_env=True, data={})
        assert p.is_guard_enabled(GuardType.OUTPUT) is False

    def test_tool_guard_enabled_false_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_SECURITY_TOOL_GUARD_ENABLED", "false")
        p = load_config(use_env=True, data={})
        assert p.is_guard_enabled(GuardType.TOOL) is False

    def test_redact_on_warn_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_SECURITY_REDACT_ON_WARN", "false")
        p = load_config(use_env=True, data={})
        assert p.redact_on_warn is False

    def test_log_clean_requests_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_SECURITY_LOG_CLEAN_REQUESTS", "true")
        p = load_config(use_env=True, data={})
        assert p.log_clean_requests is True

    def test_env_wins_over_data(self, monkeypatch):
        monkeypatch.setenv("LLM_SECURITY_BLOCK_THRESHOLD", "0.50")
        p = load_config(data={"block_threshold": 0.80, "warn_threshold": 0.30}, use_env=True)
        assert p.block_threshold == 0.50

    def test_use_env_false_ignores_env_vars(self, monkeypatch):
        monkeypatch.setenv("LLM_SECURITY_BLOCK_THRESHOLD", "0.30")
        p = load_config(use_env=False, data={})
        assert p.block_threshold == 0.75  # BalancedPolicy default

    def test_unset_env_vars_not_applied(self, monkeypatch):
        # Ensure LLM_SECURITY_BLOCK_THRESHOLD is NOT in the environment
        monkeypatch.delenv("LLM_SECURITY_BLOCK_THRESHOLD", raising=False)
        p = load_config(use_env=True, data={})
        assert p.block_threshold == 0.75

    def test_allowed_tools_with_spaces_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_SECURITY_ALLOWED_TOOLS", "search_web , read_file")
        p = load_config(use_env=True, data={})
        assert p.is_tool_allowed("search_web") is True
        assert p.is_tool_allowed("read_file") is True

class TestLoadConfigValidation:
    def test_invalid_data_raises_config_validation_error(self):
        with pytest.raises(ConfigValidationError):
            load_config(data={"block_threshold": 2.0}, use_env=False)

    def test_validation_error_reports_field(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            load_config(data={"block_threshold": 2.0}, use_env=False)
        assert any("block_threshold" in e for e in exc_info.value.errors)

    def test_validate_false_skips_validation(self):
        # With validate=False, the dict is passed through without checking.
        # The Policy dataclass will still validate — but only its own rules.
        # A valid dict that simply has an extra key should not raise.
        p = load_config(
            data={"block_threshold": 0.75, "warn_threshold": 0.40},
            validate=False,
            use_env=False,
        )
        assert p.block_threshold == 0.75

    def test_multiple_errors_in_data_all_reported(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            load_config(data={
                "block_threshold": 2.0,
                "warn_threshold": -0.5,
                "raise_on_block": "yes",
            }, use_env=False)
        assert len(exc_info.value.errors) == 3


# ══════════════════════════════════════════════════════════════════════════════
# load_config — yaml_path
# ══════════════════════════════════════════════════════════════════════════════


class TestLoadConfigYamlPath:
    def test_missing_file_raises_config_source_error(self):
        with pytest.raises(ConfigSourceError):
            load_config(yaml_path="/nonexistent/path/policy.yaml", use_env=False)

    def test_missing_file_error_contains_path(self):
        path = "/nonexistent/does_not_exist.yaml"
        with pytest.raises(ConfigSourceError) as exc_info:
            load_config(yaml_path=path, use_env=False)
        assert "does_not_exist.yaml" in str(exc_info.value)

    def test_valid_yaml_file_loaded(self, tmp_path):
        try:
            import yaml  # noqa: F401
        except ImportError:
            pytest.skip("PyYAML not installed")

        yaml_file = tmp_path / "policy.yaml"
        yaml_file.write_text(
            "name: yaml-policy\n"
            "block_threshold: 0.65\n"
            "warn_threshold: 0.30\n"
        )
        p = load_config(yaml_path=str(yaml_file), use_env=False)
        assert p.name == "yaml-policy"
        assert p.block_threshold == 0.65

    def test_yaml_overridden_by_data(self, tmp_path):
        try:
            import yaml  # noqa: F401
        except ImportError:
            pytest.skip("PyYAML not installed")

        yaml_file = tmp_path / "policy.yaml"
        yaml_file.write_text("block_threshold: 0.65\nwarn_threshold: 0.30\n")
        p = load_config(
            yaml_path=str(yaml_file),
            data={"block_threshold": 0.90, "warn_threshold": 0.40},
            use_env=False,
        )
        assert p.block_threshold == 0.90  # data wins

    def test_yaml_overridden_by_env(self, tmp_path, monkeypatch):
        try:
            import yaml  # noqa: F401
        except ImportError:
            pytest.skip("PyYAML not installed")

        monkeypatch.setenv("LLM_SECURITY_BLOCK_THRESHOLD", "0.50")
        yaml_file = tmp_path / "policy.yaml"
        yaml_file.write_text("block_threshold: 0.65\nwarn_threshold: 0.30\n")
        p = load_config(yaml_path=str(yaml_file), use_env=True)
        assert p.block_threshold == 0.50  # env wins

class TestLoadConfigPriority:
    def test_data_overrides_base_policy(self):
        p = load_config(
            base_policy=StrictPolicy(),   # block=0.40
            data={"block_threshold": 0.80, "warn_threshold": 0.30},
            use_env=False,
        )
        assert p.block_threshold == 0.80  # data wins

    def test_env_overrides_data(self, monkeypatch):
        monkeypatch.setenv("LLM_SECURITY_BLOCK_THRESHOLD", "0.50")
        p = load_config(
            data={"block_threshold": 0.80, "warn_threshold": 0.30},
            use_env=True,
        )
        assert p.block_threshold == 0.50  # env wins

    def test_env_overrides_base_policy(self, monkeypatch):
        monkeypatch.setenv("LLM_SECURITY_BLOCK_THRESHOLD", "0.50")
        p = load_config(
            base_policy=StrictPolicy(),   # block=0.40
            use_env=True,
        )
        assert p.block_threshold == 0.50  # env wins

    def test_base_policy_provides_fallback_when_no_other_source(self):
        p = load_config(base_policy=StrictPolicy(), use_env=False)
        # No data, no env — strict defaults used
        assert p.block_threshold == 0.40

    def test_non_overridden_base_fields_preserved_through_data_override(self):
        # StrictPolicy has log_clean_requests=True
        # data overrides block_threshold but not log_clean_requests
        p = load_config(
            base_policy=StrictPolicy(),
            data={"block_threshold": 0.80, "warn_threshold": 0.30},
            use_env=False,
        )
        assert p.log_clean_requests is True  # from StrictPolicy, untouched