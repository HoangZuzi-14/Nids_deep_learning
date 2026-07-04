def select_strategies(
    distribution_report,
    rare_threshold=0.001,
    focal_sampler_max_ratio=0.01,
    sampler_max_ratio=0.05,
    mild_weight_max_ratio=0.2,
):
    """Select imbalance handling per class ratio.

    GAN augmentation is intentionally not part of the default strategy. For
    tabular NIDS data, synthetic samples are hard to validate and can increase
    false alarms. The main path uses cost-sensitive learning, sampling, and rare
    grouping; GAN remains an optional research extension.
    """
    strategies = {}
    for row in distribution_report["classes"]:
        ratio = row["ratio"]
        label = row["label"]
        if ratio < rare_threshold:
            strategy = "rare_grouping"
        elif ratio < focal_sampler_max_ratio:
            strategy = "focal_loss_plus_weighted_sampler"
        elif ratio < sampler_max_ratio:
            strategy = "weighted_sampler_plus_class_weight"
        elif ratio < mild_weight_max_ratio:
            strategy = "mild_class_weight"
        else:
            strategy = "controlled_undersampling"
        strategies[str(label)] = strategy
    return strategies
