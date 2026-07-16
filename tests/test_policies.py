from __future__ import annotations
import pytest
from quisium.policies import (
    BalancedPolicy,
    GuardConfig,
    LoggingOnlyPolicy,
    Policy,
    StrictPolicy,
    load_policy_from_dict,
)
from quisium.types import GuardType, PolicyAction

class TestGuardConfigConstruction:
    def test_defaults_enabled_true(self):
        cfg = GuardConfig()
        assert cfg.enabled is True

    def test_defaults_block_threshold_none(self):
        cfg = GuardConfig()
        assert cfg.block_threshold is None

    def test_defaults_warn_threshold_none(self):
        cfg = GuardConfig()
        assert cfg.warn_threshold is None

    def test_explicit_enabled_false(self):
        cfg = GuardConfig(enabled=False)
        assert cfg.enabled is False

    def test_explicit_thresholds(self):
        cfg = GuardConfig(block_threshold=0.8, warn_threshold=0.3)
        assert cfg.block_threshold == 0.8
        assert cfg.warn_threshold == 0.3

    def test_only_block_threshold(self):
        cfg = GuardConfig(block_threshold=0.6)
        assert cfg.block_threshold == 0.6
        assert cfg.warn_threshold is None

    def test_only_warn_threshold(self):
        cfg = GuardConfig(warn_threshold=0.2)
        assert cfg.warn_threshold == 0.2
        assert cfg.block_threshold is None

    def test_boundary_block_threshold_zero(self):
        cfg = GuardConfig(block_threshold=0.0)
        assert cfg.block_threshold == 0.0

    def test_boundary_block_threshold_one(self):
        cfg = GuardConfig(block_threshold=1.0)
        assert cfg.block_threshold == 1.0


class TestGuardConfigValidation:
    def test_block_threshold_below_zero_raises(self):
        with pytest.raises(ValueError, match="block_threshold"):
            GuardConfig(block_threshold=-0.1)

    def test_block_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match="block_threshold"):
            GuardConfig(block_threshold=1.1)

    def test_warn_threshold_below_zero_raises(self):
        with pytest.raises(ValueError, match="warn_threshold"):
            GuardConfig(warn_threshold=-0.01)

    def test_warn_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match="warn_threshold"):
            GuardConfig(warn_threshold=1.01)

    def test_warn_equal_to_block_raises(self):
        with pytest.raises(ValueError, match="warn_threshold must be strictly less"):
            GuardConfig(block_threshold=0.5, warn_threshold=0.5)

    def test_warn_greater_than_block_raises(self):
        with pytest.raises(ValueError, match="warn_threshold must be strictly less"):
            GuardConfig(block_threshold=0.4, warn_threshold=0.6)

    def test_warn_strictly_less_than_block_is_valid(self):
        cfg = GuardConfig(block_threshold=0.8, warn_threshold=0.3)
        assert cfg.block_threshold == 0.8
        assert cfg.warn_threshold == 0.3

class TestPolicyConstruction:
    def test_name_default(self):
        p = Policy(block_threshold=0.75, warn_threshold=0.40)
        assert p.name == "default"

    def test_block_threshold_stored(self):
        p = Policy(name="x", block_threshold=0.80, warn_threshold=0.30)
        assert p.block_threshold == 0.80

    def test_warn_threshold_stored(self):
        p = Policy(name="x", block_threshold=0.80, warn_threshold=0.30)
        assert p.warn_threshold == 0.30

    def test_raise_on_block_default_true(self):
        p = Policy(name="x", block_threshold=0.75, warn_threshold=0.40)
        assert p.raise_on_block is True

    def test_redact_on_warn_default_true(self):
        p = Policy(name="x", block_threshold=0.75, warn_threshold=0.40)
        assert p.redact_on_warn is True

    def test_log_clean_requests_default_false(self):
        p = Policy(name="x", block_threshold=0.75, warn_threshold=0.40)
        assert p.log_clean_requests is False

    def test_allowed_tools_default_none(self):
        p = Policy(name="x", block_threshold=0.75, warn_threshold=0.40)
        assert p.allowed_tools is None

    def test_blocked_tools_default_empty_set(self):
        p = Policy(name="x", block_threshold=0.75, warn_threshold=0.40)
        assert p.blocked_tools == set()

    def test_metadata_default_empty_dict(self):
        p = Policy(name="x", block_threshold=0.75, warn_threshold=0.40)
        assert p.metadata == {}

    def test_guard_configs_default_to_enabled(self):
        p = Policy(name="x", block_threshold=0.75, warn_threshold=0.40)
        assert p.prompt_guard.enabled is True
        assert p.output_guard.enabled is True
        assert p.tool_guard.enabled is True

    def test_allowed_tools_normalised_to_lowercase(self):
        p = Policy(name="x", block_threshold=0.75, warn_threshold=0.40,
                   allowed_tools=["Search_Web", "READ_FILE"])
        assert "search_web" in p.allowed_tools
        assert "read_file" in p.allowed_tools
        assert "Search_Web" not in p.allowed_tools

    def test_blocked_tools_normalised_to_lowercase(self):
        p = Policy(name="x", block_threshold=0.75, warn_threshold=0.40,
                   blocked_tools={"DELETE_FILE"})
        assert "delete_file" in p.blocked_tools

class TestPolicyValidation:
    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            Policy(name="", block_threshold=0.75, warn_threshold=0.40)

    def test_whitespace_name_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            Policy(name="   ", block_threshold=0.75, warn_threshold=0.40)

    def test_block_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match="block_threshold"):
            Policy(name="x", block_threshold=1.5, warn_threshold=0.40)

    def test_block_threshold_below_zero_raises(self):
        with pytest.raises(ValueError, match="block_threshold"):
            Policy(name="x", block_threshold=-0.1, warn_threshold=0.0)

    def test_warn_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match="warn_threshold"):
            Policy(name="x", block_threshold=0.75, warn_threshold=1.1)

    def test_warn_threshold_below_zero_raises(self):
        with pytest.raises(ValueError, match="warn_threshold"):
            Policy(name="x", block_threshold=0.75, warn_threshold=-0.1)

    def test_warn_equal_to_block_raises(self):
        with pytest.raises(ValueError, match="warn_threshold must be strictly less"):
            Policy(name="x", block_threshold=0.5, warn_threshold=0.5)

    def test_warn_greater_than_block_raises(self):
        with pytest.raises(ValueError, match="warn_threshold must be strictly less"):
            Policy(name="x", block_threshold=0.4, warn_threshold=0.6)

    def test_valid_boundary_thresholds(self):
        # block=1.0, warn=0.0 is valid (LoggingOnly pattern)
        p = Policy(name="x", block_threshold=1.0, warn_threshold=0.0)
        assert p.block_threshold == 1.0
        assert p.warn_threshold == 0.0

class TestPresetDefaults:
    def test_balanced_name(self):
        assert BalancedPolicy().name == "balanced"

    def test_balanced_block_threshold(self):
        assert BalancedPolicy().block_threshold == 0.75

    def test_balanced_warn_threshold(self):
        assert BalancedPolicy().warn_threshold == 0.40

    def test_balanced_raise_on_block(self):
        assert BalancedPolicy().raise_on_block is True

    def test_balanced_redact_on_warn(self):
        assert BalancedPolicy().redact_on_warn is True

    def test_balanced_log_clean_requests(self):
        assert BalancedPolicy().log_clean_requests is False

    def test_strict_name(self):
        assert StrictPolicy().name == "strict"

    def test_strict_block_threshold(self):
        assert StrictPolicy().block_threshold == 0.40

    def test_strict_warn_threshold(self):
        assert StrictPolicy().warn_threshold == 0.15

    def test_strict_raise_on_block(self):
        assert StrictPolicy().raise_on_block is True

    def test_strict_redact_on_warn(self):
        assert StrictPolicy().redact_on_warn is True

    def test_strict_log_clean_requests(self):
        assert StrictPolicy().log_clean_requests is True

    def test_logging_name(self):
        assert LoggingOnlyPolicy().name == "logging-only"

    def test_logging_block_threshold(self):
        assert LoggingOnlyPolicy().block_threshold == 1.0

    def test_logging_warn_threshold(self):
        assert LoggingOnlyPolicy().warn_threshold == 0.99

    def test_logging_raise_on_block(self):
        assert LoggingOnlyPolicy().raise_on_block is False

    def test_logging_redact_on_warn(self):
        assert LoggingOnlyPolicy().redact_on_warn is False

    def test_logging_log_clean_requests(self):
        assert LoggingOnlyPolicy().log_clean_requests is True

    def test_all_presets_return_policy_instance(self):
        for factory in (BalancedPolicy, StrictPolicy, LoggingOnlyPolicy):
            assert isinstance(factory(), Policy)

class TestPresetOverrides:
    def test_balanced_custom_name(self):
        p = BalancedPolicy(name="my-policy")
        assert p.name == "my-policy"

    def test_balanced_custom_block_threshold(self):
        p = BalancedPolicy(block_threshold=0.90)
        assert p.block_threshold == 0.90

    def test_balanced_raise_on_block_false(self):
        p = BalancedPolicy(raise_on_block=False)
        assert p.raise_on_block is False

    def test_strict_custom_name(self):
        p = StrictPolicy(name="strict-prod")
        assert p.name == "strict-prod"

    def test_strict_custom_block_threshold(self):
        p = StrictPolicy(block_threshold=0.30)
        assert p.block_threshold == 0.30

    def test_logging_custom_name(self):
        p = LoggingOnlyPolicy(name="dev-logging")
        assert p.name == "dev-logging"

    def test_preset_override_allowed_tools(self):
        p = BalancedPolicy(allowed_tools=["search_web"])
        assert p.is_tool_allowed("search_web") is True
        assert p.is_tool_allowed("delete_file") is False

    def test_preset_override_blocked_tools(self):
        p = BalancedPolicy(blocked_tools={"exec_shell"})
        assert p.is_tool_allowed("exec_shell") is False

    def test_preset_override_metadata(self):
        p = BalancedPolicy(metadata={"env": "production"})
        assert p.metadata == {"env": "production"}

class TestActionForScore:
    def test_log_at_zero(self):
        assert BalancedPolicy().action_for_score(0.00, GuardType.PROMPT) == PolicyAction.LOG

    def test_log_mid_log_band(self):
        assert BalancedPolicy().action_for_score(0.20, GuardType.PROMPT) == PolicyAction.LOG

    def test_log_just_below_warn(self):
        assert BalancedPolicy().action_for_score(0.39, GuardType.PROMPT) == PolicyAction.LOG

    def test_warn_at_boundary(self):
        assert BalancedPolicy().action_for_score(0.40, GuardType.PROMPT) == PolicyAction.WARN

    def test_warn_mid_warn_band(self):
        assert BalancedPolicy().action_for_score(0.55, GuardType.PROMPT) == PolicyAction.WARN

    def test_warn_just_below_block(self):
        assert BalancedPolicy().action_for_score(0.74, GuardType.PROMPT) == PolicyAction.WARN

    def test_block_at_boundary(self):
        assert BalancedPolicy().action_for_score(0.75, GuardType.PROMPT) == PolicyAction.BLOCK

    def test_block_mid_block_band(self):
        assert BalancedPolicy().action_for_score(0.92, GuardType.PROMPT) == PolicyAction.BLOCK

    def test_block_at_one(self):
        assert BalancedPolicy().action_for_score(1.00, GuardType.PROMPT) == PolicyAction.BLOCK

    def test_strict_blocks_at_lower_boundary(self):
        assert StrictPolicy().action_for_score(0.40, GuardType.PROMPT) == PolicyAction.BLOCK

    def test_strict_blocks_medium_score(self):
        assert StrictPolicy().action_for_score(0.55, GuardType.PROMPT) == PolicyAction.BLOCK

    def test_strict_warns_below_block(self):
        assert StrictPolicy().action_for_score(0.20, GuardType.PROMPT) == PolicyAction.WARN

    def test_logging_only_never_blocks_critical(self):
        action = LoggingOnlyPolicy().action_for_score(0.92, GuardType.PROMPT)
        assert action != PolicyAction.BLOCK

    def test_logging_only_logs_clean_score(self):
        # A clean request scores 0.0, which must stay LOG, not WARN.
        action = LoggingOnlyPolicy().action_for_score(0.0, GuardType.PROMPT)
        assert action == PolicyAction.LOG

    def test_logging_only_logs_low_score(self):
        # warn_threshold=0.99, so ordinary low-risk scores stay LOG.
        action = LoggingOnlyPolicy().action_for_score(0.01, GuardType.PROMPT)
        assert action == PolicyAction.LOG

    def test_logging_only_warns_near_critical_score(self):
        # Only near-critical scores (>= 0.99) should surface a WARN.
        action = LoggingOnlyPolicy().action_for_score(0.99, GuardType.PROMPT)
        assert action == PolicyAction.WARN

    def test_all_guard_types_accepted(self):
        p = BalancedPolicy()
        for gt in GuardType:
            action = p.action_for_score(0.92, gt)
            assert action == PolicyAction.BLOCK

    def test_per_guard_override_prompt_blocks_at_lower_score(self):
        p = BalancedPolicy(prompt_guard=GuardConfig(block_threshold=0.50))
        # 0.60 is above the per-guard override of 0.50 → BLOCK
        assert p.action_for_score(0.60, GuardType.PROMPT) == PolicyAction.BLOCK

    def test_per_guard_override_tool_still_uses_policy_level(self):
        p = BalancedPolicy(prompt_guard=GuardConfig(block_threshold=0.50))
        # TOOL has no override → falls back to policy-level 0.75
        # 0.60 < 0.75 → WARN not BLOCK
        assert p.action_for_score(0.60, GuardType.TOOL) == PolicyAction.WARN

    def test_per_guard_warn_override(self):
        p = BalancedPolicy(output_guard=GuardConfig(warn_threshold=0.20))
        # 0.25 >= 0.20 (guard override) → WARN
        assert p.action_for_score(0.25, GuardType.OUTPUT) == PolicyAction.WARN
        # Without override, 0.25 < 0.40 (policy-level) → LOG
        assert BalancedPolicy().action_for_score(0.25, GuardType.OUTPUT) == PolicyAction.LOG

class TestEffectiveThresholds:
    def test_effective_block_uses_policy_level_when_no_override(self):
        p = BalancedPolicy()
        assert p.effective_block_threshold(GuardType.PROMPT) == 0.75
        assert p.effective_block_threshold(GuardType.OUTPUT) == 0.75
        assert p.effective_block_threshold(GuardType.TOOL)   == 0.75

    def test_effective_warn_uses_policy_level_when_no_override(self):
        p = BalancedPolicy()
        assert p.effective_warn_threshold(GuardType.PROMPT) == 0.40
        assert p.effective_warn_threshold(GuardType.OUTPUT) == 0.40
        assert p.effective_warn_threshold(GuardType.TOOL)   == 0.40

    def test_effective_block_uses_guard_override(self):
        p = BalancedPolicy(prompt_guard=GuardConfig(block_threshold=0.50))
        assert p.effective_block_threshold(GuardType.PROMPT) == 0.50

    def test_effective_block_other_guards_unaffected_by_prompt_override(self):
        p = BalancedPolicy(prompt_guard=GuardConfig(block_threshold=0.50))
        assert p.effective_block_threshold(GuardType.OUTPUT) == 0.75
        assert p.effective_block_threshold(GuardType.TOOL)   == 0.75

    def test_effective_warn_uses_guard_override(self):
        p = BalancedPolicy(output_guard=GuardConfig(warn_threshold=0.20))
        assert p.effective_warn_threshold(GuardType.OUTPUT) == 0.20

    def test_effective_warn_other_guards_unaffected(self):
        p = BalancedPolicy(output_guard=GuardConfig(warn_threshold=0.20))
        assert p.effective_warn_threshold(GuardType.PROMPT) == 0.40

    def test_both_overrides_on_same_guard(self):
        p = BalancedPolicy(
            tool_guard=GuardConfig(block_threshold=0.55, warn_threshold=0.20)
        )
        assert p.effective_block_threshold(GuardType.TOOL) == 0.55
        assert p.effective_warn_threshold(GuardType.TOOL)  == 0.20

class TestIsGuardEnabled:
    def test_all_enabled_by_default(self):
        p = BalancedPolicy()
        assert p.is_guard_enabled(GuardType.PROMPT) is True
        assert p.is_guard_enabled(GuardType.OUTPUT) is True
        assert p.is_guard_enabled(GuardType.TOOL)   is True

    def test_disable_prompt_guard(self):
        p = BalancedPolicy(prompt_guard=GuardConfig(enabled=False))
        assert p.is_guard_enabled(GuardType.PROMPT) is False

    def test_disable_output_guard(self):
        p = BalancedPolicy(output_guard=GuardConfig(enabled=False))
        assert p.is_guard_enabled(GuardType.OUTPUT) is False

    def test_disable_tool_guard(self):
        p = BalancedPolicy(tool_guard=GuardConfig(enabled=False))
        assert p.is_guard_enabled(GuardType.TOOL) is False

    def test_disabling_one_guard_leaves_others_enabled(self):
        p = BalancedPolicy(prompt_guard=GuardConfig(enabled=False))
        assert p.is_guard_enabled(GuardType.OUTPUT) is True
        assert p.is_guard_enabled(GuardType.TOOL)   is True

    def test_disable_all_guards(self):
        p = BalancedPolicy(
            prompt_guard=GuardConfig(enabled=False),
            output_guard=GuardConfig(enabled=False),
            tool_guard=GuardConfig(enabled=False),
        )
        for gt in GuardType:
            assert p.is_guard_enabled(gt) is False

class TestToolFiltering:
    def test_no_lists_allows_any_tool(self):
        p = BalancedPolicy()
        assert p.is_tool_allowed("search_web")  is True
        assert p.is_tool_allowed("delete_file") is True
        assert p.is_tool_allowed("exec_shell")  is True

    def test_allowlist_permits_listed_tools(self):
        p = BalancedPolicy(allowed_tools=["search_web", "read_file"])
        assert p.is_tool_allowed("search_web") is True
        assert p.is_tool_allowed("read_file")  is True

    def test_allowlist_blocks_unlisted_tools(self):
        p = BalancedPolicy(allowed_tools=["search_web"])
        assert p.is_tool_allowed("delete_file") is False
        assert p.is_tool_allowed("exec_shell")  is False

    def test_empty_allowlist_blocks_all_tools(self):
        p = BalancedPolicy(allowed_tools=[])
        assert p.is_tool_allowed("search_web") is False

    def test_blocklist_denies_listed_tools(self):
        p = BalancedPolicy(blocked_tools={"delete_file", "exec_shell"})
        assert p.is_tool_allowed("delete_file") is False
        assert p.is_tool_allowed("exec_shell")  is False

    def test_blocklist_permits_unlisted_tools(self):
        p = BalancedPolicy(blocked_tools={"delete_file"})
        assert p.is_tool_allowed("search_web") is True

    def test_blocklist_beats_allowlist_for_same_tool(self):
        p = BalancedPolicy(
            allowed_tools=["delete_file", "search_web"],
            blocked_tools={"delete_file"},
        )
        assert p.is_tool_allowed("delete_file") is False
        assert p.is_tool_allowed("search_web")  is True

    def test_allowlist_case_insensitive_mixed_case(self):
        p = BalancedPolicy(allowed_tools=["Search_Web"])
        assert p.is_tool_allowed("search_web")  is True
        assert p.is_tool_allowed("SEARCH_WEB")  is True
        assert p.is_tool_allowed("Search_Web")  is True

    def test_blocklist_case_insensitive(self):
        p = BalancedPolicy(blocked_tools={"DELETE_FILE"})
        assert p.is_tool_allowed("delete_file") is False
        assert p.is_tool_allowed("Delete_File") is False

    def test_query_case_insensitive_against_no_lists(self):
        p = BalancedPolicy()
        assert p.is_tool_allowed("SEARCH_WEB") is True

    def test_with_allowed_tools_result(self):
        p = BalancedPolicy().with_allowed_tools(["search_web"])
        assert p.is_tool_allowed("search_web")  is True
        assert p.is_tool_allowed("delete_file") is False

    def test_with_blocked_tools_result(self):
        p = BalancedPolicy().with_blocked_tools(["exec_shell"])
        assert p.is_tool_allowed("exec_shell")  is False
        assert p.is_tool_allowed("search_web")  is True

class TestPolicyMutators:
    def test_replace_returns_new_instance(self):
        p = BalancedPolicy()
        p2 = p.replace(block_threshold=0.50)
        assert p2 is not p

    def test_replace_applies_override(self):
        p = BalancedPolicy()
        p2 = p.replace(block_threshold=0.50)
        assert p2.block_threshold == 0.50

    def test_replace_does_not_mutate_original(self):
        p = BalancedPolicy()
        p.replace(block_threshold=0.50)
        assert p.block_threshold == 0.75

    def test_replace_preserves_unmentioned_fields(self):
        p = BalancedPolicy(name="base", warn_threshold=0.30)
        p2 = p.replace(block_threshold=0.60)
        assert p2.name == "base"
        assert p2.warn_threshold == 0.30

    def test_replace_name(self):
        p = BalancedPolicy()
        p2 = p.replace(name="renamed")
        assert p2.name == "renamed"

    def test_replace_raise_on_block(self):
        p = BalancedPolicy()
        p2 = p.replace(raise_on_block=False)
        assert p2.raise_on_block is False

    def test_replace_unknown_field_raises_type_error(self):
        p = BalancedPolicy()
        with pytest.raises(TypeError, match="unknown field"):
            p.replace(nonexistent_field=99)

    def test_replace_multiple_fields(self):
        p = BalancedPolicy()
        p2 = p.replace(block_threshold=0.60, warn_threshold=0.25, name="custom")
        assert p2.block_threshold == 0.60
        assert p2.warn_threshold  == 0.25
        assert p2.name            == "custom"

    def test_with_allowed_tools_returns_new_policy(self):
        p = BalancedPolicy()
        p2 = p.with_allowed_tools(["search_web"])
        assert p2 is not p

    def test_with_allowed_tools_does_not_mutate_original(self):
        p = BalancedPolicy()
        p.with_allowed_tools(["search_web"])
        assert p.allowed_tools is None

    def test_with_blocked_tools_returns_new_policy(self):
        p = BalancedPolicy()
        p2 = p.with_blocked_tools(["exec_shell"])
        assert p2 is not p

    def test_with_blocked_tools_does_not_mutate_original(self):
        p = BalancedPolicy()
        p.with_blocked_tools(["exec_shell"])
        assert p.blocked_tools == set()

class TestPolicyToDict:
    def test_contains_all_required_keys(self):
        d = BalancedPolicy().to_dict()
        expected = {
            "name", "block_threshold", "warn_threshold", "raise_on_block",
            "prompt_guard", "output_guard", "tool_guard",
            "allowed_tools", "blocked_tools", "redact_on_warn",
            "log_clean_requests", "metadata",
        }
        assert set(d.keys()) == expected

    def test_name_value(self):
        assert BalancedPolicy().to_dict()["name"] == "balanced"

    def test_block_threshold_value(self):
        assert BalancedPolicy().to_dict()["block_threshold"] == 0.75

    def test_warn_threshold_value(self):
        assert BalancedPolicy().to_dict()["warn_threshold"] == 0.40

    def test_allowed_tools_none_when_no_list(self):
        assert BalancedPolicy().to_dict()["allowed_tools"] is None

    def test_allowed_tools_list_when_set(self):
        p = BalancedPolicy(allowed_tools=["search_web"])
        assert p.to_dict()["allowed_tools"] == ["search_web"]

    def test_blocked_tools_empty_list_when_no_list(self):
        d = BalancedPolicy().to_dict()
        # blocked_tools is serialised as a sorted list
        assert isinstance(d["blocked_tools"], list)
        assert d["blocked_tools"] == []

    def test_blocked_tools_sorted_list_when_set(self):
        p = BalancedPolicy(blocked_tools={"exec_shell", "delete_file"})
        d = p.to_dict()
        assert isinstance(d["blocked_tools"], list)
        assert sorted(d["blocked_tools"]) == d["blocked_tools"]
        assert "delete_file" in d["blocked_tools"]
        assert "exec_shell"  in d["blocked_tools"]

    def test_guard_configs_serialised_as_dicts(self):
        d = BalancedPolicy().to_dict()
        for key in ("prompt_guard", "output_guard", "tool_guard"):
            assert isinstance(d[key], dict)
            assert set(d[key].keys()) == {"enabled", "block_threshold", "warn_threshold"}

    def test_guard_config_enabled_true_default(self):
        d = BalancedPolicy().to_dict()
        assert d["prompt_guard"]["enabled"] is True

    def test_guard_config_thresholds_none_when_no_override(self):
        d = BalancedPolicy().to_dict()
        assert d["prompt_guard"]["block_threshold"] is None
        assert d["prompt_guard"]["warn_threshold"]  is None

    def test_guard_config_override_serialised(self):
        p = BalancedPolicy(prompt_guard=GuardConfig(block_threshold=0.50))
        d = p.to_dict()
        assert d["prompt_guard"]["block_threshold"] == 0.50

    def test_metadata_preserved(self):
        p = BalancedPolicy(metadata={"env": "prod", "team": "platform"})
        assert p.to_dict()["metadata"] == {"env": "prod", "team": "platform"}

    def test_log_clean_requests_balanced(self):
        assert BalancedPolicy().to_dict()["log_clean_requests"] is False

    def test_log_clean_requests_strict(self):
        assert StrictPolicy().to_dict()["log_clean_requests"] is True

class TestPolicyRepr:
    def test_repr_contains_name(self):
        assert "balanced" in repr(BalancedPolicy())

    def test_repr_contains_block_threshold(self):
        assert "0.75" in repr(BalancedPolicy())

    def test_repr_contains_warn_threshold(self):
        assert "0.4" in repr(BalancedPolicy())

    def test_repr_contains_raise_on_block(self):
        assert "raise_on_block" in repr(BalancedPolicy())

    def test_repr_is_string(self):
        assert isinstance(repr(BalancedPolicy()), str)

class TestLoadPolicyFromDict:
    def test_returns_policy_instance(self):
        p = load_policy_from_dict({"block_threshold": 0.75, "warn_threshold": 0.40})
        assert isinstance(p, Policy)

    def test_name_loaded(self):
        p = load_policy_from_dict({"name": "my-policy",
                                   "block_threshold": 0.75, "warn_threshold": 0.40})
        assert p.name == "my-policy"

    def test_block_threshold_loaded(self):
        p = load_policy_from_dict({"block_threshold": 0.60, "warn_threshold": 0.25})
        assert p.block_threshold == 0.60

    def test_warn_threshold_loaded(self):
        p = load_policy_from_dict({"block_threshold": 0.60, "warn_threshold": 0.25})
        assert p.warn_threshold == 0.25

    def test_raise_on_block_false(self):
        p = load_policy_from_dict({"block_threshold": 0.75, "warn_threshold": 0.40,
                                   "raise_on_block": False})
        assert p.raise_on_block is False

    def test_allowed_tools_list_loaded(self):
        p = load_policy_from_dict({"block_threshold": 0.75, "warn_threshold": 0.40,
                                   "allowed_tools": ["search_web", "read_file"]})
        assert p.is_tool_allowed("search_web") is True
        assert p.is_tool_allowed("delete_file") is False

    def test_blocked_tools_list_converted_to_set(self):
        p = load_policy_from_dict({"block_threshold": 0.75, "warn_threshold": 0.40,
                                   "blocked_tools": ["exec_shell"]})
        assert p.is_tool_allowed("exec_shell") is False

    def test_nested_prompt_guard_dict_converted(self):
        p = load_policy_from_dict({
            "block_threshold": 0.75, "warn_threshold": 0.40,
            "prompt_guard": {"enabled": False},
        })
        assert isinstance(p.prompt_guard, GuardConfig)
        assert p.prompt_guard.enabled is False

    def test_nested_guard_block_threshold_override(self):
        p = load_policy_from_dict({
            "block_threshold": 0.75, "warn_threshold": 0.40,
            "output_guard": {"block_threshold": 0.55, "warn_threshold": 0.20},
        })
        assert p.effective_block_threshold(GuardType.OUTPUT) == 0.55
        assert p.effective_warn_threshold(GuardType.OUTPUT)  == 0.20

    def test_unknown_keys_are_silently_ignored(self):
        p = load_policy_from_dict({
            "block_threshold": 0.75, "warn_threshold": 0.40,
            "future_field": "ignored",
            "another_unknown": 42,
        })
        assert p.block_threshold == 0.75

    def test_empty_dict_returns_valid_policy(self):
        p = load_policy_from_dict({})
        assert isinstance(p, Policy)

    def test_empty_dict_uses_policy_defaults(self):
        p = load_policy_from_dict({})
        assert p.block_threshold == 0.75   # Policy dataclass default
        assert p.warn_threshold  == 0.40

    def test_metadata_loaded(self):
        p = load_policy_from_dict({
            "block_threshold": 0.75, "warn_threshold": 0.40,
            "metadata": {"env": "staging"},
        })
        assert p.metadata == {"env": "staging"}

    def test_does_not_mutate_input_dict(self):
        original = {
            "block_threshold": 0.75,
            "warn_threshold": 0.40,
            "blocked_tools": ["exec_shell"],
        }
        import copy
        original_copy = copy.deepcopy(original)
        load_policy_from_dict(original)
        assert original == original_copy