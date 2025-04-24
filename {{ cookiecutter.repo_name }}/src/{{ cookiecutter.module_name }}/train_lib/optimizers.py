"""Optimizers."""

import optax

from ..configs import default  # noqa: TID252


def create_learning_rate_schedule(config: default.Config) -> optax.Schedule:
    """Create an optax learning rate schedule."""
    schedule = config.schedule
    lr = config.learning_rate

    if schedule == "constant":
        return optax.schedules.constant_schedule(lr)
    if schedule == "exponential_decay":
        return optax.schedules.exponential_decay(
            init_value=lr,
            transition_steps=config.transition_steps,
            decay_rate=config.decay_rate,
        )
    if schedule == "warmup_exponential_decay":
        return optax.schedules.warmup_exponential_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=config.warmup_steps,
            transition_steps=config.transition_steps,
            decay_rate=config.decay_rate,
        )
    if schedule == "cosine_decay":
        return optax.schedules.cosine_decay_schedule(
            init_value=lr, decay_steps=config.num_train_steps
        )
    if schedule == "warmup_cosine_decay":
        return optax.schedules.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=config.warmup_steps,
            decay_steps=config.decay_steps,
        )
    # Unknown learning rate schedule.
    msg = "Invalid schedule: " + repr(schedule)
    raise ValueError(msg)


def create_optimizer(
    config: default.Config, learning_rate_schedule: optax.Schedule
) -> optax.GradientTransformation:
    """Create an optax optimizer."""
    if config.optimizer == "adam":
        return optax.adam(learning_rate_schedule)
    if config.optimizer == "adamw":
        return optax.adamw(learning_rate_schedule, weight_decay=config.weight_decay)
    if config.optimizer == "sgd":
        return optax.sgd(learning_rate_schedule)
    msg = "Unsupported optimizer"
    raise ValueError(msg)
