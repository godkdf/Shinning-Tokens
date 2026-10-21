# Security and service scope

The current maintained publication candidate is **0.2.3**. Report the exact release and input/artifact identities when describing a problem.

The Python service binds to **loopback only**. It checks Host/Origin, limits request bodies and validates known paths/selections. It is a local desktop service, not an authenticated internet API. Publish the static `site/` explorer rather than exposing this Python service publicly.

Benchmark/conversion requests use cached inputs. Downloads run only after an explicit selection/action. Custom import accepts tokenizer JSON data and does not execute custom Python. Reference wrapper source retained beside some tokenizer assets is inspection data and is never imported by the lab. No model weights are needed.

Portable kits verify pinned runtimes/native wheels and include an offline guard. Hash checks establish integrity against the recorded manifest, not trust in an arbitrary replacement manifest or proof of execution on another OS. Native Windows/Linux/Intel Mac checks remain unverified until their manual workflow is actually run.

Do not put access tokens, personal documents, model weights, `.env` files or private local paths in public issue reports or release assets. Use GitHub's private vulnerability reporting if it is available on the repository; otherwise contact the repository owner through an existing private channel. Do not post exploit details or credentials in a public issue. No email address or response-time guarantee is implied.

Upstream dataset and tokenizer licensing is documented separately in [THIRD_PARTY.md](THIRD_PARTY.md); source-code licensing does not replace those terms.
