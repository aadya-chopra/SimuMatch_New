import matplotlib.pyplot as plt
import sys, os

# Path from pages/ → SimuMatch-main/src/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC_PATH = os.path.join(ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


def plot_bar_chart(df):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(df["sport"], df["normalized_fit"])
    ax.set_ylabel("Normalized Fit Score")
    ax.set_title("Sport Compatibility Ranking")
    plt.xticks(rotation=45)
    return fig
