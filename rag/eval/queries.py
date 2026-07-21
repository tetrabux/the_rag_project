EVAL_QUERIES = [
    # identifier: unique config key, exactly one correct file
    {"query": "What does MAX_AUTH_IDLE_TIMEOUT control?", "tag": "identifier", "expected_files": ["err_4182.md"]},
    {"query": "What is MAX_INGEST_IDLE_TIMEOUT?", "tag": "identifier", "expected_files": ["err_4210.md"]},
    {"query": "MAX_AUTH_BATCH_SIZE", "tag": "identifier", "expected_files": ["err_4259.md"]},

    # conceptual: natural language, manual docs
    {"query": "How do I install the software?", "tag": "conceptual", "expected_files": ["getting_started.md"]},
    {"query": "What happens the first time I start the service?", "tag": "conceptual", "expected_files": ["getting_started.md"]},
    {"query": "How does the system scale?", "tag": "conceptual", "expected_files": ["operations.md"]},
    {"query": "What is the consistency model of the system?", "tag": "conceptual", "expected_files": ["architecture.md"]},
    {"query": "How do I back up my data?", "tag": "conceptual", "expected_files": ["operations.md"]},

    # multi_section: answer spans more than one section/file
    {"query": "How do authentication and multi-tenancy interact in this system?", "tag": "multi_section", "expected_files": ["security.md"]},
    {"query": "What should I know about scaling and monitoring the worker tier?", "tag": "multi_section", "expected_files": ["operations.md"]},
    {"query": "What is the architecture and consistency model of the system?", "tag": "multi_section", "expected_files": ["architecture.md"]},

    # filtered: meant to be run with a source filter
    {"query": "retry backoff configuration", "tag": "filtered", "filter": {"source": "wiki"}, "expected_files": None},
    {"query": "installation", "tag": "filtered", "filter": {"source": "manual"}, "expected_files": ["getting_started.md"]},
    {"query": "How does authentication get configured?", "tag": "filtered", "filter": {"source": "manual"}, "expected_files": ["security.md"]},

    # unanswerable: nothing in corpus should match
    {"query": "How do I configure Kubernetes autoscaling for this service?", "tag": "unanswerable", "expected_files": []},
    {"query": "What is the pricing model for the enterprise tier?", "tag": "unanswerable", "expected_files": []},
    {"query": "How do I integrate this with Slack notifications?", "tag": "unanswerable", "expected_files": []},

    # misspelled: typo'd version of a real query
    {"query": "MAX_NETWROK_RETRY_BACKOFF", "tag": "misspelled", "expected_files": ["err_4000.md", "err_4161.md", "err_4336.md", "err_4364.md"]},
    {"query": "How do I instal the sofware?", "tag": "misspelled", "expected_files": ["getting_started.md"]},

    # near_duplicate: identifier shared across multiple near-identical docs
    {"query": "MAX_NETWORK_RETRY_BACKOFF", "tag": "near_duplicate", "expected_files": ["err_4000.md", "err_4161.md", "err_4336.md", "err_4364.md"]},
    {"query": "MAX_AUTH_RETRY_BACKOFF", "tag": "near_duplicate", "expected_files": ["err_4021.md", "err_4070.md", "err_4238.md"]},
    {"query": "MAX_STORAGE_BATCH_SIZE", "tag": "near_duplicate", "expected_files": ["err_4098.md", "err_4147.md", "err_4224.md", "err_4252.md", "err_4308.md"]},
    {"query": "What controls the storage batch size limit?", "tag": "near_duplicate", "expected_files": ["err_4098.md", "err_4147.md", "err_4224.md", "err_4252.md", "err_4308.md"]},
]
