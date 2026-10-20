"""Unit tests for OrchestratorAgent, focusing on delegation workflows and QA findings."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from finbot.agents.orchestrator import OrchestratorAgent
from finbot.core.auth.session import SessionContext


@pytest.fixture(autouse=True)
def mock_event_bus():
    """Mock event_bus emission to avoid network/Redis dependencies in unit tests."""
    with (
        patch(
            "finbot.core.messaging.event_bus.emit_agent_event",
            new_callable=AsyncMock,
        ),
        patch(
            "finbot.core.messaging.event_bus.emit_business_event",
            new_callable=AsyncMock,
        ),
    ):
        yield


@pytest.fixture
def session_context():
    now = datetime.now(UTC)
    return SessionContext(
        session_id="sess-orch-001",
        user_id="test-user-orch",
        is_temporary=False,
        namespace="default",
        created_at=now,
        expires_at=now + timedelta(hours=1),
    )


@pytest.fixture
def orchestrator_agent(session_context):
    return OrchestratorAgent(
        session_context=session_context,
        workflow_id="wf-test-orch-001",
    )


class TestQAFindings:
    """Regression tests for Orchestrator QA audit findings."""

    @pytest.mark.asyncio
    async def test_orch_qa_002_next_step_on_failed_payment_misleads_llm(self, orchestrator_agent):
        """When payments agent fails, next_step must not instruct payment_confirmation."""
        failed_payment_result = {
            "task_status": "failed",
            "task_summary": "Payment declined — insufficient funds.",
        }

        with patch(
            "finbot.agents.runner.run_payments_agent",
            new_callable=AsyncMock,
            return_value=failed_payment_result,
        ):
            result = await orchestrator_agent.delegate_to_payments(
                invoice_id=1,
                task_description="Process vendor invoice payment",
            )

        assert result["task_status"] == "failed"
        assert "next_step" not in result

    @pytest.mark.asyncio
    async def test_delegate_to_payments_completed_includes_next_step(self, orchestrator_agent):
        """When payments agent succeeds with completed, next_step is attached."""
        success_result = {
            "task_status": "completed",
            "task_summary": "Payment settled successfully via Stripe.",
        }

        with patch(
            "finbot.agents.runner.run_payments_agent",
            new_callable=AsyncMock,
            return_value=success_result,
        ):
            result = await orchestrator_agent.delegate_to_payments(
                invoice_id=1,
                task_description="Process vendor invoice payment",
            )

        assert result["task_status"] == "completed"
        assert "next_step" in result
        assert "payment_confirmation" in result["next_step"]

    @pytest.mark.asyncio
    async def test_delegate_to_payments_success_status_includes_next_step(self, orchestrator_agent):
        """When payments agent succeeds with success, next_step is attached."""
        success_result = {
            "task_status": "success",
            "task_summary": "Payment completed.",
        }

        with patch(
            "finbot.agents.runner.run_payments_agent",
            new_callable=AsyncMock,
            return_value=success_result,
        ):
            result = await orchestrator_agent.delegate_to_payments(
                invoice_id=1,
                task_description="Process vendor invoice payment",
            )

        assert result["task_status"] == "success"
        assert "next_step" in result
        assert "payment_confirmation" in result["next_step"]

    @pytest.mark.asyncio
    async def test_delegate_to_payments_missing_status_no_next_step(self, orchestrator_agent):
        """When payments agent result has no task_status, next_step must not be added."""
        ambiguous_result = {
            "task_summary": "Incomplete response without status",
        }

        with patch(
            "finbot.agents.runner.run_payments_agent",
            new_callable=AsyncMock,
            return_value=ambiguous_result,
        ):
            result = await orchestrator_agent.delegate_to_payments(
                invoice_id=1,
                task_description="Process vendor invoice payment",
            )

        assert "next_step" not in result
