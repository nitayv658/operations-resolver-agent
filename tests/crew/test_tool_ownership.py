"""Static separation-of-concerns assertions -- the kit's own suggested check
(starter-kit README: "assert 'process_refund' not in {t['name'] for t in
mat.COMMS_TOOLS}"), enforced as a real test rather than a comment, and
extended to check the crew's own agent wiring, not just the kit's bundles.
"""

from __future__ import annotations

from resolver_agent.crew.comms.agent import CommsAgent
from resolver_agent.crew.decision.agent import DecisionAgent
from resolver_agent.crew.researcher.agent import ResearcherAgent


def test_kit_bundles_partition_the_tool_set(mat_module):
    mat = mat_module
    names = {t["name"] for t in mat.RESEARCHER_TOOLS} | {t["name"] for t in mat.DECISION_TOOLS} | {
        t["name"] for t in mat.COMMS_TOOLS
    }
    assert names == set(mat.TOOL_REGISTRY)
    assert "process_refund" not in {t["name"] for t in mat.COMMS_TOOLS}
    assert "process_refund" not in {t["name"] for t in mat.RESEARCHER_TOOLS}
    assert "send_slack_alert" not in {t["name"] for t in mat.RESEARCHER_TOOLS}
    assert "send_slack_alert" not in {t["name"] for t in mat.DECISION_TOOLS}


def test_researcher_agent_registry_matches_its_role(mat_module):
    agent = ResearcherAgent(client=object(), model="x")
    assert set(agent.tool_registry) == {"get_order_details", "get_user_profile", "audit_fraud_risk"}
    for name in agent.tool_registry:
        assert mat_module.TOOL_OWNERSHIP[name] == "researcher"


def test_decision_agent_registry_matches_its_role(mat_module):
    agent = DecisionAgent(client=object(), model="x")
    assert set(agent.tool_registry) == {"check_return_policy", "process_refund"}
    for name in agent.tool_registry:
        assert mat_module.TOOL_OWNERSHIP[name] == "decision"


def test_comms_agent_registry_matches_its_role_and_cannot_refund(mat_module):
    agent = CommsAgent(client=object(), model="x")
    assert set(agent._base_tool_registry) == {"get_escalation_route", "send_slack_alert"}
    for name in agent._base_tool_registry:
        assert mat_module.TOOL_OWNERSHIP[name] == "comms"
    assert "process_refund" not in agent._base_tool_registry
