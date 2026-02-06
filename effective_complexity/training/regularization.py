# effective_complexity/training/regularization.py

def elastic_net_regularization(model, cfg):
    """
    Elastic-net regularization:
      l1_weight * l1_lambda * ||θ||_1
    + l2_weight * l2_lambda * ||θ||_2^2
    """
    model_cfg = cfg.get("model", {})
    l1_weight = model_cfg.get("l1_weight", 0.0)
    l2_weight = model_cfg.get("l2_weight", 0.0)

    l1_lambda = model_cfg.get("l1_lambda", 1e-4)
    l2_lambda = model_cfg.get("l2_lambda", 1e-4)

    l1_term = sum(p.abs().sum() for p in model.parameters())
    l2_term = sum((p ** 2).sum() for p in model.parameters())

    return (
        l1_weight * l1_lambda * l1_term
        + l2_weight * l2_lambda * l2_term
    )
