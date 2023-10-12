
__all__ = ["params_freeze", "save_results", "save_preds"]


def params_freeze(model, para_update="klg"):
    """
    Divide params into with-weight-decay and without-weight-decay groups.
    Layernorms and baises will have no weight decay but the rest will.
    """
    
    print("=" * 20 + "all parameters name" + "=" * 20)
    for name, param in model.named_parameters():
        print(name)
        pass

    print("=" * 20 + "require updating name" + "=" * 20)
    """Freeze all the parameters except knowledge related module."""
    for name, param in model.named_parameters():
        if para_update not in name:
            param.requires_grad = False
            pass
        else:
            print(name)
            pass
        pass
    

    return model


def save_results(results_file_path, results):
    with open(results_file_path, 'w') as f:
        for k, v in results.items():
            f.write(f"{k}={v}\n")


def save_preds(preds_file_path, ids_preds):
    with open(preds_file_path, 'w') as f:
        ids, preds = ids_preds
        for id, pred in zip(ids, preds):
            f.write(f"{id},{pred}\n")
