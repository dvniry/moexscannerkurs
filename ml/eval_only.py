# ml/eval_only.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == '__main__':
    import torch
    from ml.config import CFG
    from ml.dataset_v3 import build_full_multiscale_dataset_v3, temporal_split, INDICATOR_COLS
    from ml.multiscale_cnn_v3 import MultiScaleHybridV3, evaluate_multiscale_v3
    from ml.visualize_predictions import predict_and_plot
    from torch.utils.data import Subset

    MODEL_PATH = 'ml/model_multiscale_v3.pt'

    dataset, y_all, ctx_dim, ticker_lengths = build_full_multiscale_dataset_v3(
        force_rebuild=False, use_hourly=True
    )
    _, _, idx_test = temporal_split(ticker_lengths)
    y_test = y_all[idx_test]
    te_ds  = Subset(dataset, idx_test.tolist())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = MultiScaleHybridV3(
        ctx_dim=ctx_dim,
        n_indicator_cols=len(INDICATOR_COLS),
        future_bars=CFG.future_bars,
        use_hourly=True,
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))

    evaluate_multiscale_v3(model, te_ds, y_test, ctx_dim,
                           use_hourly=True,
                           save_json=MODEL_PATH.replace('.pt', '_eval.json'))

    predict_and_plot(MODEL_PATH, te_ds, y_test, ctx_dim,
                     use_hourly=True, n_examples=8)