"""Webhook verification utilities for VLMRun."""

import hmac
import hashlib
from typing import Optional, Union


def verify_webhook(
    raw_body: Union[str, bytes], signature_header: Optional[str], secret: str
) -> bool:
    """
    Verify webhook HMAC signature.

    This function verifies that a webhook request came from VLM Run by validating
    the HMAC signature in the X-VLM-Signature header. The signature is computed
    using SHA256 HMAC with your webhook secret.

    Args:
        raw_body: Raw request body as string or bytes
        signature_header: X-VLM-Signature header value (format: "sha256=<hex>")
        secret: Your webhook secret from VLM Run dashboard

    Returns:
        True if the signature is valid, False otherwise

    Example:
        ```python
        from fastapi import FastAPI, Request, HTTPException
        from vlmrun.common.webhook import verify_webhook

        app = FastAPI()

        @app.post("/webhook")
        async def webhook_endpoint(request: Request):
            raw_body = await request.body()
            signature_header = request.headers.get("X-VLM-Signature")
            secret = "your_webhook_secret_here"

            if not verify_webhook(raw_body, signature_header, secret):
                raise HTTPException(status_code=401, detail="Invalid signature")

            import json
            data = json.loads(raw_body)
            return {"status": "success"}
        ```
    """
    if not signature_header or not signature_header.startswith("sha256="):
        return False

    if not secret:
        return False

    received_sig = signature_header.replace("sha256=", "")

    body_bytes = raw_body.encode("utf-8") if isinstance(raw_body, str) else raw_body

    expected_sig = hmac.new(
        secret.encode("utf-8"), body_bytes, hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(received_sig, expected_sig)
