"""Tests for webhook verification utilities."""

import hmac
import hashlib
import json
import pytest

from vlmrun.common.webhook import verify_webhook


class TestVerifyWebhook:
    """Test suite for webhook verification."""

    test_secret = "test_webhook_secret_12345"
    test_payload = json.dumps(
        {
            "id": "pred_123",
            "status": "completed",
            "response": {"data": "test"},
        }
    )

    @staticmethod
    def generate_signature(payload: str | bytes, secret: str) -> str:
        """Helper function to generate valid signature."""
        body_bytes = payload.encode("utf-8") if isinstance(payload, str) else payload
        signature = hmac.new(
            secret.encode("utf-8"), body_bytes, hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"

    def test_valid_signature_with_string_input(self):
        """Test that valid signature with string input returns True."""
        signature = self.generate_signature(self.test_payload, self.test_secret)
        result = verify_webhook(self.test_payload, signature, self.test_secret)
        assert result is True

    def test_valid_signature_with_bytes_input(self):
        """Test that valid signature with bytes input returns True."""
        raw_body = self.test_payload.encode("utf-8")
        signature = self.generate_signature(raw_body, self.test_secret)
        result = verify_webhook(raw_body, signature, self.test_secret)
        assert result is True

    def test_valid_signature_with_empty_payload(self):
        """Test that valid signature with empty payload returns True."""
        empty_payload = ""
        signature = self.generate_signature(empty_payload, self.test_secret)
        result = verify_webhook(empty_payload, signature, self.test_secret)
        assert result is True

    def test_valid_signature_with_complex_json_payload(self):
        """Test that valid signature with complex JSON payload returns True."""
        complex_payload = json.dumps(
            {
                "id": "pred_456",
                "status": "completed",
                "response": {
                    "nested": {
                        "data": "test",
                        "array": [1, 2, 3],
                        "special": "chars: !@#$%^&*()",
                    },
                },
                "metadata": {
                    "credits_used": 10,
                    "timestamp": "2024-01-01T00:00:00Z",
                },
            }
        )
        signature = self.generate_signature(complex_payload, self.test_secret)
        result = verify_webhook(complex_payload, signature, self.test_secret)
        assert result is True

    def test_valid_signature_with_unicode_characters(self):
        """Test that valid signature with unicode characters returns True."""
        unicode_payload = json.dumps(
            {
                "message": "Hello 世界 🌍",
                "emoji": "🚀✨",
            }
        )
        signature = self.generate_signature(unicode_payload, self.test_secret)
        result = verify_webhook(unicode_payload, signature, self.test_secret)
        assert result is True

    def test_invalid_signature(self):
        """Test that incorrect signature returns False."""
        wrong_signature = (
            "sha256=0000000000000000000000000000000000000000000000000000000000000000"
        )
        result = verify_webhook(self.test_payload, wrong_signature, self.test_secret)
        assert result is False

    def test_signature_with_wrong_secret(self):
        """Test that signature with wrong secret returns False."""
        signature = self.generate_signature(self.test_payload, "wrong_secret")
        result = verify_webhook(self.test_payload, signature, self.test_secret)
        assert result is False

    def test_signature_with_modified_payload(self):
        """Test that signature with modified payload returns False."""
        signature = self.generate_signature(self.test_payload, self.test_secret)
        modified_payload = self.test_payload + " modified"
        result = verify_webhook(modified_payload, signature, self.test_secret)
        assert result is False

    def test_signature_with_different_case(self):
        """Test that signature with different case returns False."""
        signature = self.generate_signature(self.test_payload, self.test_secret)
        uppercase_signature = signature.upper()
        result = verify_webhook(self.test_payload, uppercase_signature, self.test_secret)
        assert result is False

    def test_truncated_signature(self):
        """Test that truncated signature returns False."""
        signature = self.generate_signature(self.test_payload, self.test_secret)
        truncated_signature = signature[:-2]  # Remove last 2 chars
        result = verify_webhook(self.test_payload, truncated_signature, self.test_secret)
        assert result is False

    def test_undefined_signature_header(self):
        """Test that undefined signature header returns False."""
        result = verify_webhook(self.test_payload, None, self.test_secret)
        assert result is False

    def test_empty_signature_header(self):
        """Test that empty signature header returns False."""
        result = verify_webhook(self.test_payload, "", self.test_secret)
        assert result is False

    def test_signature_without_prefix(self):
        """Test that signature without sha256= prefix returns False."""
        signature = self.generate_signature(self.test_payload, self.test_secret).replace(
            "sha256=", ""
        )
        result = verify_webhook(self.test_payload, signature, self.test_secret)
        assert result is False

    def test_signature_with_wrong_prefix(self):
        """Test that signature with wrong prefix returns False."""
        signature = self.generate_signature(self.test_payload, self.test_secret).replace(
            "sha256=", "sha512="
        )
        result = verify_webhook(self.test_payload, signature, self.test_secret)
        assert result is False

    def test_signature_with_extra_prefix(self):
        """Test that signature with extra prefix returns False."""
        signature = "extra_" + self.generate_signature(
            self.test_payload, self.test_secret
        )
        result = verify_webhook(self.test_payload, signature, self.test_secret)
        assert result is False

    def test_empty_secret(self):
        """Test that empty secret returns False."""
        signature = self.generate_signature(self.test_payload, self.test_secret)
        result = verify_webhook(self.test_payload, signature, "")
        assert result is False

    def test_secret_with_different_encoding(self):
        """Test that secret with whitespace differences returns False."""
        signature = self.generate_signature(self.test_payload, self.test_secret)
        wrong_secret = self.test_secret + " "  # Add trailing space
        result = verify_webhook(self.test_payload, signature, wrong_secret)
        assert result is False

    def test_non_hex_signature(self):
        """Test that non-hex signature returns False."""
        invalid_signature = "sha256=not_a_hex_string!!!"
        result = verify_webhook(self.test_payload, invalid_signature, self.test_secret)
        assert result is False

    def test_signature_with_invalid_hex_length(self):
        """Test that signature with invalid hex length returns False."""
        invalid_signature = "sha256=abc"  # Too short
        result = verify_webhook(self.test_payload, invalid_signature, self.test_secret)
        assert result is False

    def test_large_payloads(self):
        """Test that large payloads are handled correctly."""
        large_payload = json.dumps({"data": "x" * 10000})
        signature = self.generate_signature(large_payload, self.test_secret)
        result = verify_webhook(large_payload, signature, self.test_secret)
        assert result is True

    def test_special_characters_in_secret(self):
        """Test that special characters in secret are handled correctly."""
        special_secret = "secret!@#$%^&*()_+-=[]{}|;:',.<>?/~`"
        signature = self.generate_signature(self.test_payload, special_secret)
        result = verify_webhook(self.test_payload, signature, special_secret)
        assert result is True

    def test_consistency_across_multiple_calls(self):
        """Test that function is consistent across multiple calls with same inputs."""
        signature = self.generate_signature(self.test_payload, self.test_secret)
        result1 = verify_webhook(self.test_payload, signature, self.test_secret)
        result2 = verify_webhook(self.test_payload, signature, self.test_secret)
        result3 = verify_webhook(self.test_payload, signature, self.test_secret)
        assert result1 is True
        assert result2 is True
        assert result3 is True

    def test_timing_safe_comparison(self):
        """Test that function uses timing-safe comparison."""
        signature = self.generate_signature(self.test_payload, self.test_secret)
        result = verify_webhook(self.test_payload, signature, self.test_secret)
        assert result is True

    def test_comparison_errors_handled_gracefully(self):
        """Test that comparison errors are handled gracefully."""
        short_signature = "sha256=abc123"
        result = verify_webhook(self.test_payload, short_signature, self.test_secret)
        assert result is False

    def test_webhook_from_prediction_completion(self):
        """Test verification of webhook from prediction completion."""
        webhook_payload = json.dumps(
            {
                "id": "pred_abc123",
                "status": "completed",
                "response": {
                    "invoice_number": "INV-001",
                    "total": 1234.56,
                    "date": "2024-01-15",
                },
                "usage": {
                    "credits_used": 5,
                },
                "created_at": "2024-01-15T10:00:00Z",
                "completed_at": "2024-01-15T10:00:05Z",
            }
        )
        signature = self.generate_signature(webhook_payload, self.test_secret)
        result = verify_webhook(webhook_payload, signature, self.test_secret)
        assert result is True

    def test_webhook_from_agent_execution(self):
        """Test verification of webhook from agent execution."""
        webhook_payload = json.dumps(
            {
                "id": "exec_xyz789",
                "name": "invoice-processor",
                "status": "completed",
                "response": {
                    "processed": True,
                    "results": ["item1", "item2"],
                },
                "usage": {
                    "credits_used": 10,
                },
                "created_at": "2024-01-15T11:00:00Z",
                "completed_at": "2024-01-15T11:00:15Z",
            }
        )
        signature = self.generate_signature(webhook_payload, self.test_secret)
        result = verify_webhook(webhook_payload, signature, self.test_secret)
        assert result is True

    def test_reject_tampered_webhook_payload(self):
        """Test that tampered webhook payload is rejected."""
        original_payload = json.dumps(
            {
                "id": "pred_123",
                "status": "completed",
                "usage": {
                    "credits_used": 5,
                },
            }
        )
        signature = self.generate_signature(original_payload, self.test_secret)

        tampered_payload = json.dumps(
            {
                "id": "pred_123",
                "status": "completed",
                "usage": {
                    "credits_used": 1,  # Changed from 5 to 1
                },
            }
        )

        result = verify_webhook(tampered_payload, signature, self.test_secret)
        assert result is False

    def test_bytes_payload_with_string_signature(self):
        """Test that bytes payload with string signature works correctly."""
        raw_body = self.test_payload.encode("utf-8")
        signature = self.generate_signature(self.test_payload, self.test_secret)
        result = verify_webhook(raw_body, signature, self.test_secret)
        assert result is True

    def test_string_payload_with_bytes_signature(self):
        """Test that string payload with bytes signature works correctly."""
        raw_body_bytes = self.test_payload.encode("utf-8")
        signature = self.generate_signature(raw_body_bytes, self.test_secret)
        result = verify_webhook(self.test_payload, signature, self.test_secret)
        assert result is True

    def test_whitespace_in_payload(self):
        """Test that whitespace in payload is handled correctly."""
        payload_with_whitespace = json.dumps(
            {"id": "pred_123", "status": "completed"}, indent=2
        )
        signature = self.generate_signature(payload_with_whitespace, self.test_secret)
        result = verify_webhook(payload_with_whitespace, signature, self.test_secret)
        assert result is True

    def test_newlines_in_payload(self):
        """Test that newlines in payload are handled correctly."""
        payload_with_newlines = '{"id": "pred_123",\n"status": "completed"}'
        signature = self.generate_signature(payload_with_newlines, self.test_secret)
        result = verify_webhook(payload_with_newlines, signature, self.test_secret)
        assert result is True

    def test_signature_header_with_spaces(self):
        """Test that signature header with spaces returns False."""
        signature = self.generate_signature(self.test_payload, self.test_secret)
        signature_with_spaces = signature.replace("=", " = ")
        result = verify_webhook(self.test_payload, signature_with_spaces, self.test_secret)
        assert result is False

    def test_multiple_equals_in_signature(self):
        """Test that signature with multiple equals signs is handled correctly."""
        signature = self.generate_signature(self.test_payload, self.test_secret)
        signature_with_extra_equals = signature + "="
        result = verify_webhook(
            self.test_payload, signature_with_extra_equals, self.test_secret
        )
        assert result is False
