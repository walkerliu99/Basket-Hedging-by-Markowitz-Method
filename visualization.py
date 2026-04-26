import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def render_table(df, title):
    fig, ax = plt.subplots(figsize=(10, len(df) * 0.6 + 1.5))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", color="white", pad=20)

    table = ax.table(
        cellText=df.round(6).values,
        colLabels=df.columns,
        rowLabels=df.index,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    for _, cell in table.get_celld().items():
        cell.set_edgecolor("gray")
        cell.set_text_props(color="white")
        cell.set_facecolor("#1e1e1e")

    plt.tight_layout()
    plt.show()


def plot_cumulative_returns(cum_returns, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    cum_returns.plot(ax=ax, linewidth=1.5)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_lambda_sweep(
    cum_lam,
    title="Cumulative Returns for Different Lambda Values",
    ylim=None,
):
    fig, ax = plt.subplots(figsize=(12, 6))

    cum_lam.plot(ax=ax, linewidth=1.5)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")

    if ylim is not None:
        ylim = (-0.6, 0.6)

    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    plt.xticks(rotation=45)
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.show()

def plot_strategy_comparison(cum_returns, title):
    fig, ax = plt.subplots(figsize=(12, 6))

    cum_returns.plot(ax=ax, linewidth=1.5)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    plt.xticks(rotation=45)
    plt.legend(loc="best", fontsize=10)
    plt.tight_layout()
    plt.show()