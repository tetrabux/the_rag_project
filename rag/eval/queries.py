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

    # paraphrase: correct chunk shares no literal terms with the query (forces semantic match, not string match)
    {"query": "If too many jobs are running at once and slowing things down, what are the two opposite adjustments I could try, and what should I check before adding more capacity?",
     "tag": "paraphrase",
     "expected_files": ["err_4021.md", "err_4049.md", "err_4140.md", "err_4154.md", "err_4266.md", "err_4287.md", "err_4301.md", "err_4315.md", "err_4322.md", "err_4329.md", "err_4385.md"]},
    {"query": "What's the safe way to handle two processes updating the same piece of data at the same time, and why is blindly retrying a bad idea?",
     "tag": "paraphrase",
     "expected_files": ["err_4007.md", "err_4042.md", "err_4105.md", "err_4119.md", "err_4133.md", "err_4168.md", "err_4217.md", "err_4231.md", "err_4273.md", "err_4336.md", "err_4357.md", "err_4364.md"]},

    # distractor: resolution text shared by ~19 docs whose titles/descriptions are independently randomized and mostly unrelated
    {"query": "How do I fix repeated signing key failures — should I rotate immediately?",
     "tag": "distractor",
     "expected_files": ["err_4035.md", "err_4056.md", "err_4070.md", "err_4098.md", "err_4112.md", "err_4126.md", "err_4147.md", "err_4175.md", "err_4189.md", "err_4196.md", "err_4238.md", "err_4245.md", "err_4252.md", "err_4259.md", "err_4280.md", "err_4308.md", "err_4343.md", "err_4378.md", "err_4399.md"]},

    # near_duplicate_precision: 10 docs share this exact description, but only ONE is auth-subsystem — returning any of the other 9 near-identical docs instead is wrong
    {"query": "For the auth-subsystem error whose description mentions a signing key not found in the active keyring, what config key controls it and what's the default?",
     "tag": "near_duplicate_precision",
     "expected_files": ["err_4259.md"]},
]
