import evaluate_modular_results


def test_rejects_degenerate_one_class_model_result():
    metrics = {
        "confusion_matrix": [[146]],
        "classification_report": {
            "0": {"precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 146.0},
            "accuracy": 1.0,
        },
    }

    valid, reason = evaluate_modular_results.is_valid_model_evaluation(metrics)

    assert valid is False
    assert reason == "degenerate_one_class"


def test_accepts_multiclass_model_result_with_test_support():
    metrics = {
        "confusion_matrix": [[9, 1], [2, 8]],
        "classification_report": {
            "0": {"precision": 0.8, "recall": 0.9, "f1-score": 0.85, "support": 10.0},
            "1": {"precision": 0.9, "recall": 0.8, "f1-score": 0.84, "support": 10.0},
            "accuracy": 0.85,
        },
    }

    valid, reason = evaluate_modular_results.is_valid_model_evaluation(metrics)

    assert valid is True
    assert reason == ""
