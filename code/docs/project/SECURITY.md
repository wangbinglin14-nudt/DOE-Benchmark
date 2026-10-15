# Security Policy

`doe-testbed` is a research-grade scientific computing library — it does not
process untrusted user data, run web services in production, or grant
elevated privileges.  Nevertheless, security disclosures are welcome.

## Supported versions

| Version | Status |
|---|---|
| `0.4.x` (current `main`) | ✅ Active development; security fixes land here. |
| `0.3.x` | ⚠️ Critical security fixes only; otherwise frozen. |
| `< 0.3` | ❌ End-of-life. |

## Reporting a vulnerability

**Please do not open a public Issue.**  Use either of the following
private channels:

1. **Email** (current channel): `wangbinglin14 [at] nudt [dot] edu [dot] cn`.
   Please do not include credentials or other sensitive material in the first
   message; a secure exchange method can be arranged if needed.
2. **GitHub Security Advisory**: this channel will be enabled after the
   current release is published to its public repository.

When reporting, please include:

- Affected file(s) / version(s)
- A minimal reproducer (Python snippet or Docker command)
- The threat model you have in mind (e.g. arbitrary-code execution
  through a malicious case ID, denial of service via an infinite
  evaluation loop, sandbox escape inside the Docker image)
- Suggested fix, if any

## Response timeline

| Step | Target |
|---|---|
| Initial acknowledgement | within 72 h |
| Triage and severity assessment | within 7 days |
| Fix and coordinated release | within 30 days for high-severity issues |
| Public disclosure (CVE) | once a fix is available; credit reporters by name unless they prefer anonymity |

## Threat model and explicit non-goals

The library **is not** designed to:

- Sandbox the execution of untrusted Python code.  Loading arbitrary
  Python modules through `importlib` or `pickle` should be assumed
  to grant the same privileges as the calling Python process.
- Run as a multi-tenant service.  The web dashboard
  (`web/backend`) is a developer convenience and does not implement
  authentication or rate-limiting.  Do not expose it to the public
  internet.
- Defend against supply-chain attacks on its own dependencies.  We
  pin versions in `requirements.lock` and rebuild the Docker image
  nightly via the `Benchmark / docker` CI job to surface upstream
  drift, but trust in `numpy`, `scipy`, `pytorch`, `botorch` etc.
  is inherited.

## Hardening recommendations for users

- Run the benchmark inside the provided Docker image
  (`doe-bench:0.4.0`).  The image runs under a non-root user
  `bench`, has no compilers in the runtime layer, and bind-mounts
  `benchmark/results/` read-write only.
- Use `requirements.lock` (Python 3.13.12) for byte-exact
  reproducibility.  Drift in scientific dependencies has historically
  produced silent numerical changes; the `determinism` CI job
  detects this on every PR.
- Verify the Zenodo deposit's checksum against the public release tag SHA
  before running on shared infrastructure.

## Disclosure history

No vulnerabilities have been reported as of this release.

---

Last reviewed: 2026-09-01 (v0.4.0-prereg candidate).
