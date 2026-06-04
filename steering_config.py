def parse_steering_specs(args):
    if getattr(args, "steer", None):
        specs = {}
        for raw_spec in args.steer:
            layer_idx, feature_indices = _parse_one_steer_spec(raw_spec)
            specs.setdefault(layer_idx, []).extend(feature_indices)
        return specs

    feature_idx = getattr(args, "feature_idx", None)
    if not feature_idx:
        raise ValueError("At least one feature is required via --steer or --feature_idx.")
    layer_idx = int(args.layer_idx)
    features = [int(feature) for feature in feature_idx]
    if layer_idx < 0 or any(feature < 0 for feature in features):
        raise ValueError("Layer and feature indices must be non-negative.")
    return {layer_idx: features}


def format_steering_specs(specs):
    parts = []
    for layer_idx in sorted(specs):
        features = "_".join(str(feature_idx) for feature_idx in specs[layer_idx])
        parts.append(f"L{layer_idx}-{features}")
    return "__".join(parts)


def _parse_one_steer_spec(raw_spec):
    if ":" not in raw_spec:
        raise ValueError(f"Invalid --steer value '{raw_spec}'. Expected LAYER:FEATURE,FEATURE.")

    raw_layer, raw_features = raw_spec.split(":", 1)
    if not raw_layer.strip() or not raw_features.strip():
        raise ValueError(f"Invalid --steer value '{raw_spec}'. Expected LAYER:FEATURE,FEATURE.")

    try:
        layer_idx = int(raw_layer)
    except ValueError as exc:
        raise ValueError(f"Invalid layer in --steer value '{raw_spec}'.") from exc
    if layer_idx < 0:
        raise ValueError(f"Invalid layer in --steer value '{raw_spec}'.")

    feature_indices = []
    for raw_feature in raw_features.split(","):
        raw_feature = raw_feature.strip()
        if not raw_feature:
            raise ValueError(f"Invalid feature in --steer value '{raw_spec}'.")
        try:
            feature_idx = int(raw_feature)
        except ValueError as exc:
            raise ValueError(f"Invalid feature in --steer value '{raw_spec}'.") from exc
        if feature_idx < 0:
            raise ValueError(f"Invalid feature in --steer value '{raw_spec}'.")
        feature_indices.append(feature_idx)

    return layer_idx, feature_indices
