from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from src.pipeline.dl_runner import run_dl_experiment


MODEL_DISPLAY_NAMES = {
    "mlp": "MLP",
    "cnn1d": "CNN1D",
    "bilstm": "BiLSTMAttention",
    "hybrid": "CNNLSTMHybrid",
}


def merge_results_to_unified(dataset_key: str, classification_type: str, experiment_name: str, model_name: str) -> None:
    results_dir = ROOT / "results"
    individual_path = results_dir / f"{dataset_key}_{classification_type}_{experiment_name}_results.json"
    unified_path = results_dir / f"{dataset_key}_{classification_type}_modular_results.json"

    if not individual_path.exists():
        print(f"[-] Individual results file not found: {individual_path}")
        return

    # Load individual results
    with individual_path.open("r", encoding="utf-8") as f:
        ind_data = json.load(f)

    metrics = ind_data["models"][experiment_name]

    # Load unified results if exists, else initialize
    if unified_path.exists():
        with unified_path.open("r", encoding="utf-8") as f:
            uni_data = json.load(f)
    else:
        print(f"[*] Unified results file not found, creating a new one: {unified_path}")
        uni_data = {
            "dataset": dataset_key,
            "classification": classification_type,
            "label_mapping": ind_data.get("label_mapping", {}),
            "models": {}
        }

    # Save under standard display name (e.g., CNN1D instead of cnn1d or MLP_FocalSampler)
    display_name = MODEL_DISPLAY_NAMES.get(model_name.lower(), model_name)
    uni_data["models"][display_name] = metrics

    # Save back to unified file
    with unified_path.open("w", encoding="utf-8") as f:
        json.dump(uni_data, f, indent=2)
    print(f"[+] Merged metrics for {display_name} into {unified_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Automation Suite to run all Deep Learning models.")
    parser.add_argument("--smoke-test", action="store_true", help="Run short epochs on small fraction of NSL-KDD for verification.")
    parser.add_argument("--dataset", choices=["nsl_kdd", "unsw_nb15", "cicids2017"], help="Run a specific dataset.")
    parser.add_argument("--model", choices=["mlp", "cnn1d", "bilstm", "hybrid"], help="Run a specific model.")
    parser.add_argument("--sample-fraction", type=float, help="Override sample fraction.")
    parser.add_argument("--epochs", type=int, help="Override epochs.")
    parser.add_argument("--batch-size", type=int, help="Override batch size.")
    parser.add_argument("--lr", type=float, help="Override learning rate.")
    parser.add_argument("--patience", type=int, help="Override early stopping patience.")
    args = parser.parse_args()

    # Determine datasets and models to run
    if args.smoke_test:
        datasets = ["nsl_kdd"]
        models = ["mlp", "cnn1d", "bilstm", "hybrid"]
        sample_frac = args.sample_fraction if args.sample_fraction is not None else 1.0
        epochs = args.epochs if args.epochs is not None else 1
        patience = args.patience if args.patience is not None else 1
        print("[*] Running Smoke Test with sample fraction 1.0, 1 epoch for all models.")
    else:
        datasets = [args.dataset] if args.dataset else ["nsl_kdd", "unsw_nb15", "cicids2017"]
        models = [args.model] if args.model else ["mlp", "cnn1d", "bilstm", "hybrid"]
        sample_frac = args.sample_fraction if args.sample_fraction is not None else 1.0
        epochs = args.epochs
        patience = args.patience

    classification_type = "multi"

    # Execution matrix
    for ds in datasets:
        print(f"\n========================================\nDataset: {ds.upper()}\n========================================")
        for model in models:
            # Set dynamic experiment name
            display_name = MODEL_DISPLAY_NAMES.get(model.lower(), model)
            experiment_name = f"{display_name}_Modular"
            if args.smoke_test:
                experiment_name = f"{display_name}_Smoke"

            print(f"\n[*] Training model: {display_name} (Experiment: {experiment_name})")
            
            try:
                # Call run_dl_experiment directly
                run_dl_experiment(
                    dataset_key=ds,
                    classification_type=classification_type,
                    root=ROOT,
                    experiment_name=experiment_name,
                    model_name=model,
                    use_focal_loss=False,
                    use_weighted_sampler=False,
                    sample_fraction=sample_frac,
                    epochs_override=epochs,
                    batch_size_override=args.batch_size,
                    lr_override=args.lr,
                    patience_override=patience,
                )
                print(f"[+] Successfully trained {display_name} on {ds}.")

                # Merge individual results into unified dataset results
                merge_results_to_unified(
                    dataset_key=ds,
                    classification_type=classification_type,
                    experiment_name=experiment_name,
                    model_name=model,
                )

            except Exception as e:
                print(f"[-] ERROR training {display_name} on {ds}: {e}")
                if args.smoke_test:
                    # Reraise exception for visibility during smoke testing
                    raise e

    # Update global modular results summary and CSVs by calling evaluate_modular_results.py
    print("\n[*] Re-evaluating modular results summary and CSVs...")
    try:
        subprocess.run([sys.executable, str(ROOT / "evaluate_modular_results.py")], check=True)
        print("[+] Re-evaluation completed successfully!")
    except Exception as e:
        print(f"[-] Error calling evaluate_modular_results.py: {e}")


if __name__ == "__main__":
    main()
