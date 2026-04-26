from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
UNIVERSE_DIR = BASE_DIR / "data" / "universe"


def ensure_universe_dir() -> None:
    UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)


def clean_ticker(ticker: str) -> str:
    """
    Convert raw ticker symbols into Yahoo Finance style.

    Examples:
    BRK.B -> BRK-B
    BRK/B -> BRK-B
    """
    if pd.isna(ticker):
        return ""

    ticker = str(ticker).strip().upper()
    ticker = ticker.replace(".", "-")
    ticker = ticker.replace("/", "-")

    return ticker


def load_symbol_file(filename: str, asset_type: str) -> pd.DataFrame:
    """
    Load one universe CSV from data/universe/.

    The file must contain a column named Symbol.
    """
    ensure_universe_dir()

    path = UNIVERSE_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Universe file not found: {path}")

    df = pd.read_csv(path)

    # Allow case-insensitive matching, but expect Symbol
    col_map = {c.lower(): c for c in df.columns}

    if "symbol" not in col_map:
        raise ValueError(
            f"{filename} must contain a column named Symbol. "
            f"Columns found: {list(df.columns)}"
        )

    symbol_col = col_map["symbol"]

    out = pd.DataFrame()
    out["symbol_raw"] = df[symbol_col]
    out["ticker"] = df[symbol_col].apply(clean_ticker)
    out["asset_type"] = asset_type
    out["source_file"] = filename

    # Keep optional name/security columns if they exist
    for possible_name_col in ["Name", "Security Name", "Description"]:
        if possible_name_col in df.columns:
            out["name"] = df[possible_name_col]
            break

    out = out[out["ticker"] != ""]
    out = out.drop_duplicates(subset=["ticker"])

    return out


def load_csv_symbol_universe(
    etf_file: str = "core_etfs.csv",
    stock_file: str = "custom_universe.csv",
    target: str | None = None,
    benchmark: str | None = None,
    include_target: bool = False,
    include_benchmark: bool = False,
    save_files: bool = True,
) -> pd.DataFrame:
    """
    Load ETF and stock universes from local CSV files.

    Both files must contain a Symbol column.

    Assumption:
    - etf_file contains ETF symbols
    - stock_file contains stock symbols
    """
    ensure_universe_dir()

    etfs = load_symbol_file(etf_file, asset_type="ETF")
    stocks = load_symbol_file(stock_file, asset_type="equity")

    universe = pd.concat([etfs, stocks], ignore_index=True)

    universe["ticker"] = universe["ticker"].apply(clean_ticker)
    universe = universe[universe["ticker"] != ""]
    universe = universe.drop_duplicates(subset=["ticker"])

    if target is not None and not include_target:
        universe = universe[universe["ticker"] != clean_ticker(target)]

    if benchmark is not None and not include_benchmark:
        universe = universe[universe["ticker"] != clean_ticker(benchmark)]

    universe = universe.sort_values(["asset_type", "ticker"]).reset_index(drop=True)

    if save_files:
        output_path = UNIVERSE_DIR / "csv_symbol_universe_clean.csv"
        universe.to_csv(output_path, index=False)
        print(f"Saved cleaned CSV universe to: {output_path}")

    return universe


def get_ticker_list(universe_df: pd.DataFrame) -> list[str]:
    return universe_df["ticker"].dropna().astype(str).tolist()


if __name__ == "__main__":
    universe_df = load_csv_symbol_universe(
        etf_file="core_etfs.csv",
        stock_file="custom_universe.csv",
        target="AAPL",
        benchmark="SPY",
        include_target=False,
        include_benchmark=False,
        save_files=True,
    )

    tickers = get_ticker_list(universe_df)

    print("Candidate universe size:", len(tickers))
    print("First 25 tickers:", tickers[:25])
    print("AAPL in universe?", "AAPL" in tickers)
    print("SPY in universe?", "SPY" in tickers)
    print("\nAsset type counts:")
    print(universe_df["asset_type"].value_counts())