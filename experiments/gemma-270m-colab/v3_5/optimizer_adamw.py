def make_optimizer(config, params):
    schedule = optax.join_schedules([
        optax.linear_schedule(0., config["learning_rate"], config["warmup_updates"]),
        optax.constant_schedule(config["learning_rate"]),
    ], [config["warmup_updates"]])
    decay = jax.tree.map(lambda value: value.ndim > 1, params)
    transform = optax.chain(
        optax.clip_by_global_norm(config["gradient_clip"]),
        optax.adamw(schedule, b1=.9, b2=.999, eps=1e-8,
                    weight_decay=config["weight_decay"], mask=decay))
    flat = traverse_util.flatten_dict(params)
    count = sum(value.size for value in flat.values())
    routing = [{"path": "/".join(path), "shape": list(value.shape),
                "parameters": value.size, "optimizer": "AdamW",
                "weight_decay": value.ndim > 1}
               for path, value in sorted(flat.items())]
    return transform, {"name": "AdamW", "adamw_parameters": count,
        "muon_parameters": 0, "learning_rate": config["learning_rate"],
        "warmup_updates": config["warmup_updates"], "routing": routing}
