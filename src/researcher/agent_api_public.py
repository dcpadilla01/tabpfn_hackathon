"""Public names of agent_api (kept dependency-free so the static checker can import it cheaply)."""

PUBLIC_API = frozenset({
    "RESEARCH_VISIBLE_DAY", "AsOf", "snapshot", "history", "build_features", "baseline_features",
    "train_targets", "snapshot_days", "save_table", "load_saved", "describe_tables", "KEYS", "TARGET", "pd", "np",
})
