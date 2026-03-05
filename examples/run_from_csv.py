import pandas as pd

from mabt import mabt_ci

def main():
    true_labels = pd.read_csv("examples/data/labels.csv", header=None)
    pred_labels = pd.read_csv("examples/data/predictions.csv", header=None)

    bound, tau, t0 = mabt_ci(true_labels, pred_labels)

    print(f"MABT lower bound: {bound:.6f}")
    print(f"Tilting parameter: {tau:.6f}")
    print(f"Point estimate (optimistic!): {t0:.6f}")

if __name__ == "__main__":
    main()