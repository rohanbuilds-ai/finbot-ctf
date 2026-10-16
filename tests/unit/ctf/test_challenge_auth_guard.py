"""
Unit tests for CTF challenge endpoint authentication guards.

Verifies that state-modifying CTF challenge endpoints (check_challenge, use_hint)
strictly enforce persistent authenticated sessions (get_authenticated_session_context),
preventing anonymous or temporary guest sessions from bypassing attempt tracking
or consuming hints.
"""

import json

import pytest
from fastapi.testclient import TestClient

from finbot.config import settings
from finbot.core.auth.session import session_manager
from finbot.core.data.models import Challenge


@pytest.fixture
def sample_challenge(db):
    """Seed an active challenge with hints for testing."""
    challenge = Challenge(
        id="chal-auth-guard-test",
        title="Auth Guard Test Challenge",
        description="Verifies authentication guard on check and hint endpoints",
        category="prompt_injection",
        subcategory="system_prompt_leak",
        difficulty="beginner",
        points=100,
        detector_class="SystemPromptLeakDetector",
        hints=json.dumps(
            [
                {"text": "First hint for test", "cost": 10},
                {"text": "Second hint for test", "cost": 20},
            ]
        ),
        is_active=True,
        order_index=1,
    )
    db.add(challenge)
    db.commit()
    db.refresh(challenge)
    return challenge


@pytest.mark.unit
class TestChallengeAuthGuard:
    """Test authentication enforcement on CTF challenge endpoints."""

    def test_anonymous_session_check_challenge_rejected(
        self, fast_client: TestClient, sample_challenge: Challenge
    ):
        """Temporary guest sessions cannot invoke check_challenge and receive 401."""
        # Initialize client session via GET
        fast_client.get("/")
        session_id = fast_client.cookies.get(settings.SESSION_COOKIE_NAME)
        temp_session, _ = session_manager.get_session(session_id)
        assert temp_session is not None
        assert temp_session.is_temporary is True

        headers = {
            settings.CSRF_HEADER_NAME: temp_session.csrf_token,
            "Accept": "application/json",
        }
        response = fast_client.post(
            f"/ctf/api/v1/challenges/{sample_challenge.id}/check",
            headers=headers,
        )

        assert response.status_code == 401

    def test_anonymous_session_use_hint_rejected(
        self, fast_client: TestClient, sample_challenge: Challenge
    ):
        """Temporary guest sessions cannot consume hints and receive 401."""
        fast_client.get("/")
        session_id = fast_client.cookies.get(settings.SESSION_COOKIE_NAME)
        temp_session, _ = session_manager.get_session(session_id)
        assert temp_session is not None
        assert temp_session.is_temporary is True

        headers = {
            settings.CSRF_HEADER_NAME: temp_session.csrf_token,
            "Accept": "application/json",
        }
        response = fast_client.post(
            f"/ctf/api/v1/challenges/{sample_challenge.id}/hint",
            headers=headers,
        )

        assert response.status_code == 401

    def test_authenticated_session_check_challenge_allowed(
        self, fast_client: TestClient, sample_challenge: Challenge
    ):
        """Authenticated sessions can invoke check_challenge and receive CheckResult."""
        auth_session = session_manager.create_session(
            email="authenticated_hacker@example.com",
            user_agent="testclient",
            ip_address="testclient",
        )
        assert auth_session.is_temporary is False

        fast_client.cookies.set(settings.SESSION_COOKIE_NAME, auth_session.session_id)
        headers = {
            settings.CSRF_HEADER_NAME: auth_session.csrf_token,
            "Accept": "application/json",
        }

        response = fast_client.post(
            f"/ctf/api/v1/challenges/{sample_challenge.id}/check",
            headers=headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("available", "in_progress", "completed")
        assert "detected" in data
        assert "confidence" in data
        assert "message" in data

    def test_authenticated_session_use_hint_allowed(
        self, fast_client: TestClient, sample_challenge: Challenge
    ):
        """Authenticated sessions can use hints and receive HintResponse."""
        auth_session = session_manager.create_session(
            email="authenticated_hacker@example.com",
            user_agent="testclient",
            ip_address="testclient",
        )
        assert auth_session.is_temporary is False

        fast_client.cookies.set(settings.SESSION_COOKIE_NAME, auth_session.session_id)
        headers = {
            settings.CSRF_HEADER_NAME: auth_session.csrf_token,
            "Accept": "application/json",
        }

        response = fast_client.post(
            f"/ctf/api/v1/challenges/{sample_challenge.id}/hint",
            headers=headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["hint_index"] == 0
        assert data["hint_text"] == "First hint for test"
        assert data["points_deducted"] == 10
        assert data["total_hints_cost"] == 10

    def test_authenticated_session_nonexistent_challenge_returns_404(
        self, fast_client: TestClient, db
    ):
        """Authenticated sessions querying non-existent challenge receive 404."""
        auth_session = session_manager.create_session(
            email="authenticated_hacker@example.com",
            user_agent="testclient",
            ip_address="testclient",
        )
        fast_client.cookies.set(settings.SESSION_COOKIE_NAME, auth_session.session_id)
        headers = {
            settings.CSRF_HEADER_NAME: auth_session.csrf_token,
            "Accept": "application/json",
        }

        response = fast_client.post(
            "/ctf/api/v1/challenges/nonexistent-chal-xyz/check",
            headers=headers,
        )
        assert response.status_code == 404

    def test_public_challenge_read_endpoints_remain_accessible(
        self, fast_client: TestClient, sample_challenge: Challenge
    ):
        """Public read endpoints (list, get) remain accessible to unauthenticated/guest sessions."""
        # Anonymous listing
        response_list = fast_client.get("/ctf/api/v1/challenges")
        assert response_list.status_code == 200
        challenges = response_list.json()
        assert isinstance(challenges, list)
        assert any(c["id"] == sample_challenge.id for c in challenges)

        # Anonymous getting details
        response_get = fast_client.get(f"/ctf/api/v1/challenges/{sample_challenge.id}")
        assert response_get.status_code == 200
        detail = response_get.json()
        assert detail["id"] == sample_challenge.id
        assert detail["title"] == sample_challenge.title
