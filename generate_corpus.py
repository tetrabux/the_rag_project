"""Generates the synthetic corpus. No learning value here - just run it once.

Produces docs/ with two sources:
  - manual/    : product manual, prose-heavy, few rare identifiers
  - wiki/      : error-code wiki, dense with rare identifiers (ERR_xxxx, CONFIG_KEYS)

The split matters: dense retrieval will do fine on manual/ and fall over on wiki/.
That contrast is the whole point of the corpus.
"""
import random
from pathlib import Path

random.seed(42)

OUT = Path(__file__).parent / "docs"

SUBSYSTEMS = ["network", "storage", "auth", "scheduler", "ingest"]

ERROR_TEMPLATE = """## ERR_{code} — {title}

**Subsystem:** {subsystem}
**Severity:** {severity}
**Introduced in:** v{version}

### Description

{description}

### Common causes

{causes}

### Resolution

{resolution}

### Related configuration

The behaviour of this error is governed by `{config_key}`, which defaults to
`{default}`. Increasing this value will {effect}. See the configuration
reference for interaction with `{related_key}`.
"""

DESCRIPTIONS = [
    "The client terminated the connection before the server finished streaming the response body.",
    "A write was rejected because the target segment had already been sealed by a concurrent compaction.",
    "Token validation failed because the signing key referenced in the header was not present in the active keyring.",
    "The task was evicted from the queue after exceeding its allotted wall-clock budget.",
    "An upstream fetch returned a payload whose declared content-length did not match the bytes received.",
    "The replica refused the request because its view of the cluster epoch was stale.",
    "A batch was dropped because the in-memory buffer reached capacity before the flush interval elapsed.",
]

CAUSES = [
    "- Aggressive client-side timeouts under sustained load\n- Proxy buffering limits set below the response size\n- Packet loss on the path between edge and origin",
    "- Two writers targeting the same segment without coordination\n- A compaction job scheduled during peak write volume\n- Clock skew between coordinator nodes",
    "- Key rotation completed without a grace window\n- The keyring cache was not invalidated after rotation\n- A stale replica serving an expired key set",
    "- Under-provisioned worker pool\n- A single task monopolising a shared executor\n- Backpressure from a downstream dependency",
]

RESOLUTIONS = [
    "Increase the client read timeout and confirm the proxy is not buffering the full body. If the error persists under low load, inspect the network path rather than the timeout configuration.",
    "Serialise writes to a given segment through a single coordinator, or enable optimistic retry. Do not simply retry blindly, as this amplifies the contention that caused the failure.",
    "Re-issue the token against the current keyring. If rotation is in progress, allow a grace window of at least two token lifetimes before retiring the previous key.",
    "Raise the worker pool size or lower the per-task budget. Confirm the downstream dependency is healthy before increasing concurrency, as additional workers will otherwise queue behind the same bottleneck.",
]

EFFECTS = [
    "reduce spurious failures at the cost of holding connections open longer",
    "increase memory pressure on the coordinator",
    "delay detection of genuinely stuck tasks",
    "improve throughput but widen the window for duplicate delivery",
]

MANUAL_SECTIONS = [
    ("Getting Started", "Installation", "Install the runtime using the package manager for your platform. The service expects a writable data directory and a configuration file at the default path. On first start the service will initialise its metadata store, which may take several minutes on large volumes."),
    ("Getting Started", "First Run", "After installation, start the service and confirm it reports healthy. The readiness endpoint will return a non-ready status until the metadata store has finished initialising. Do not send production traffic before readiness is reported."),
    ("Architecture", "Overview", "The system is composed of a coordinator tier and a worker tier. Coordinators own metadata and scheduling decisions; workers execute tasks and own no durable state. This separation allows workers to be replaced freely without data migration."),
    ("Architecture", "Consistency Model", "Writes are acknowledged once a quorum of replicas has durably accepted them. Reads may be served from any replica and are therefore eventually consistent unless a linearisable read is explicitly requested. Applications that require read-your-writes semantics should pin reads to the coordinator."),
    ("Operations", "Scaling", "The worker tier scales horizontally. Adding workers increases aggregate throughput but does not reduce the latency of any individual task. Where per-task latency matters, reduce the task size rather than adding capacity."),
    ("Operations", "Backup and Restore", "Backups capture the metadata store and segment index. Restoring to a cluster with a different topology requires a rebalance pass, which the restore tool will perform automatically but which may take considerable time proportional to data volume."),
    ("Operations", "Monitoring", "Export metrics to your collector of choice. The most operationally useful signals are queue depth, flush latency, and replica lag. Alerting on CPU alone is discouraged, as the system is typically bound by IO rather than compute."),
    ("Security", "Authentication", "All requests must present a bearer token. Tokens are signed with a key from the active keyring and carry a tenant claim which scopes every subsequent operation."),
    ("Security", "Multi-tenancy", "Tenant isolation is enforced at the metadata layer. Every query is rewritten to include a tenant predicate before it reaches the storage engine. Application code must never rely on filtering results after retrieval, as this places correctness in the wrong layer."),
]


def build_error_docs():
    d = OUT / "wiki"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(60):
        code = 4000 + i * 7
        sub = random.choice(SUBSYSTEMS)
        content = ERROR_TEMPLATE.format(
            code=code,
            title=random.choice(DESCRIPTIONS).split(".")[0][:60],
            subsystem=sub,
            severity=random.choice(["warning", "error", "fatal"]),
            version=f"{random.randint(1,4)}.{random.randint(0,9)}",
            description=random.choice(DESCRIPTIONS),
            causes=random.choice(CAUSES),
            resolution=random.choice(RESOLUTIONS),
            config_key=f"MAX_{sub.upper()}_{random.choice(['RETRY_BACKOFF','BUFFER_BYTES','IDLE_TIMEOUT','BATCH_SIZE'])}",
            default=random.choice(["30s", "512MB", "1000", "5m"]),
            effect=random.choice(EFFECTS),
            related_key=f"{sub.upper()}_{random.choice(['POOL_SIZE','FLUSH_INTERVAL','QUORUM_SIZE'])}",
        )
        (d / f"err_{code}.md").write_text(f"# Error Reference\n\n{content}")
    print(f"wrote 60 wiki docs -> {d}")


def build_manual():
    d = OUT / "manual"
    d.mkdir(parents=True, exist_ok=True)
    by_chapter = {}
    for chapter, section, body in MANUAL_SECTIONS:
        by_chapter.setdefault(chapter, []).append((section, body))
    for chapter, sections in by_chapter.items():
        parts = [f"# {chapter}\n"]
        for section, body in sections:
            # padded out so sections are long enough to force sub-splitting
            padded = (body + " ") * 6
            parts.append(f"\n## {section}\n\n{padded.strip()}\n")
        slug = chapter.lower().replace(" ", "_")
        (d / f"{slug}.md").write_text("\n".join(parts))
    print(f"wrote {len(by_chapter)} manual docs -> {d}")


if __name__ == "__main__":
    build_error_docs()
    build_manual()
