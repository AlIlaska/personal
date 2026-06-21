"""
train_model.py
==============
Fit the GameRatingPipeline on your cleaned dataset and save it as a pickle that
the Streamlit app loads. Run this once locally (where Final_clean_df.csv lives):

    python train_model.py --data Final_clean_df.csv --out game_rating_model.pkl

Options:
    --model {lasso,ridge,elasticnet}   default: lasso  (best in the notebook)
    --test-size FLOAT                  default: 0.2    (time-based holdout)
"""

import argparse
import pickle

import pandas as pd

from pipeline import GameRatingPipeline


def main():
    ap = argparse.ArgumentParser(description="Train Steam game-rating model.")
    ap.add_argument("--data", required=True, help="Path to cleaned CSV (e.g. Final_clean_df.csv)")
    ap.add_argument("--out", default="game_rating_model.pkl", help="Output pickle path")
    ap.add_argument("--model", default="lasso", choices=["lasso", "ridge", "elasticnet"])
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    print(f"Loading {args.data} ...")
    df = pd.read_csv(args.data)
    print(f"  {len(df):,} rows, {df.shape[1]} columns")

    pipe = GameRatingPipeline(model_type=args.model)
    pipe.fit(df, test_size=args.test_size, verbose=True)

    with open(args.out, "wb") as f:
        pickle.dump(pipe, f)
    print(f"\nSaved model -> {args.out}")
    print("You can now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
