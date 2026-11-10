# Security policy

## Credential handling

- Never commit `.env`, provider API keys, access tokens, model artifacts, or private audio datasets.
- Keep `OPENAI_API_KEY` and `TYPECAST_API_KEY` on the backend only. Flutter receives only the backend URL.
- `TYPECAST_VOICE_ID` must be a valid voice ID from the Typecast API voice library.
- Credentials previously committed to repository history must be revoked and rotated at the provider; deleting a string from a newer commit does not invalidate it or remove historical Git objects.

## Deployment

- Use HTTPS and a trusted reverse proxy/API gateway.
- Add real user or device authentication before exposing the backend publicly. The in-process IP throttle is a basic local safeguard, not production-grade abuse prevention.
- Store model artifacts in a trusted, access-controlled location. `joblib` artifacts are Python pickle-based and must never be loaded from an untrusted source.
- Keep uploaded audio and generated speech in temporary storage and remove them after processing.

## Reporting

Do not publish exploitable vulnerabilities or secret values in public issues. Use the repository's private vulnerability reporting feature when it is enabled.
