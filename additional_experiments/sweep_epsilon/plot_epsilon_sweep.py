import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Plot delta vs mismatch score from a CSV file.")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file with 'delta' and 'mismatch_score' columns.")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path (default: same directory as CSV, with .png extension).")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    df = pd.read_csv(csv_path)

    # Support both standard and segmental sweep result schemas.
    if "mismatch_score" in df.columns:
        y_col = "mismatch_score"
        y_label = "reconstruction error"
    elif "seg_mismatch_mean" in df.columns:
        y_col = "seg_mismatch_mean"
        y_label = "segmental reconstruction error"
    else:
        raise ValueError(
            "CSV must contain either 'mismatch_score' or 'seg_mismatch_mean' column."
        )

    fig, ax = plt.subplots(figsize=(6, 4))
    if y_col == "seg_mismatch_mean" and "seg_mismatch_std" in df.columns:
        ax.errorbar(
            df["epsilon"],
            df[y_col],
            yerr=df["seg_mismatch_std"],
            marker="o",
            linestyle="-",
            capsize=3,
        )
    else:
        ax.plot(df["epsilon"], df[y_col], marker="o")

    if csv_path.stem.endswith("1holes"):
        title_context = "simulation"
    elif csv_path.stem.endswith("Gardner"):
        title_context = "2D experimental data"
    else:
        title_context = csv_path.stem

    ax.set_xlabel(r"epsilon($\epsilon$)")
    ax.set_ylabel(y_label)
    ax.set_title(f"Epsilon vs {y_label}\n({title_context})")
    fig.tight_layout()

    out = Path(args.output) if args.output else csv_path.with_suffix(".png")
    #fig.savefig(out, dpi=150)
    pdf_out = out.with_suffix(".pdf")
    fig.savefig(pdf_out)
    plt.close(fig)
    #print(f"Saved {out}")
    print(f"Saved {pdf_out}")


if __name__ == "__main__":
    main()
