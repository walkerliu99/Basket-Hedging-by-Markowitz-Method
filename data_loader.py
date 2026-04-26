import time
import hashlib
from pathlib import Path

import pandas as pd
import yfinance as yf


def make_price_cache_path(
    tickers,
    target,
    benchmark,
    start_date,
    end_date,
    price_dir="data/prices",
):
    price_dir = Path(price_dir)
    price_dir.mkdir(parents=True, exist_ok=True)

    universe_hash = hashlib.md5("|".join(tickers).encode()).hexdigest()[:10]

    return (
        price_dir
        / f"raw_yfinance_{target}_{benchmark}_{start_date}_{end_date}_{universe_hash}.pkl"
    )


def batch_download_yfinance(
    tickers,
    start_date,
    end_date,
    batch_size=100,
    pause_seconds=10,
    auto_adjust=False,
):
    all_batches = []
    successful_tickers = []
    failed_tickers = []

    total = len(tickers)

    for start_idx in range(0, total, batch_size):
        batch = tickers[start_idx:start_idx + batch_size]
        batch_num = start_idx // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        print(f"\nDownloading batch {batch_num}/{total_batches}")
        print(f"Tickers {start_idx + 1} to {min(start_idx + batch_size, total)} out of {total}")

        try:
            batch_raw = yf.download(
                batch,
                start=start_date,
                end=end_date,
                auto_adjust=auto_adjust,
                group_by="column",
                progress=False,
                threads=True,
            )

            if batch_raw.empty:
                print("Batch returned empty data.")
                failed_tickers.extend(batch)
                continue

            if isinstance(batch_raw.columns, pd.MultiIndex):
                if "Close" in batch_raw.columns.get_level_values(0):
                    close_batch = batch_raw["Close"]

                    if isinstance(close_batch, pd.Series):
                        close_batch = close_batch.to_frame(name=batch[0])

                    batch_success = close_batch.columns[close_batch.notna().any()].tolist()
                else:
                    batch_success = []
            else:
                if "Close" in batch_raw.columns and batch_raw["Close"].notna().any():
                    batch_success = batch
                else:
                    batch_success = []

            batch_failed = [t for t in batch if t not in batch_success]

            successful_tickers.extend(batch_success)
            failed_tickers.extend(batch_failed)
            all_batches.append(batch_raw)

            print(f"Successful in batch: {len(batch_success)}")
            print(f"Failed in batch: {len(batch_failed)}")

        except Exception as e:
            print(f"Batch failed with error: {e}")
            failed_tickers.extend(batch)

        time.sleep(pause_seconds)

    if len(all_batches) == 0:
        raise RuntimeError("No data downloaded from yfinance.")

    raw = pd.concat(all_batches, axis=1)
    raw = raw.loc[:, ~raw.columns.duplicated()]

    successful_tickers = sorted(list(set(successful_tickers)))
    failed_tickers = sorted(list(set(failed_tickers)))

    print("\nBatch download complete.")
    print("Requested tickers:", len(tickers))
    print("Successful tickers:", len(successful_tickers))
    print("Failed tickers:", len(failed_tickers))

    return raw, successful_tickers, failed_tickers


def load_or_download_prices(
    tickers,
    target,
    benchmark,
    start_date,
    end_date,
    batch_size=100,
    pause_seconds=10,
    auto_adjust=False,
):
    price_cache_path = make_price_cache_path(
        tickers=tickers,
        target=target,
        benchmark=benchmark,
        start_date=start_date,
        end_date=end_date,
    )

    if price_cache_path.exists():
        print(f"Loading cached price data from: {price_cache_path}")
        raw = pd.read_pickle(price_cache_path)

        close_tmp = raw["Close"]
        successful_tickers = close_tmp.columns[close_tmp.notna().any()].tolist()
        failed_tickers = [t for t in tickers if t not in successful_tickers]

    else:
        raw, successful_tickers, failed_tickers = batch_download_yfinance(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            batch_size=batch_size,
            pause_seconds=pause_seconds,
            auto_adjust=auto_adjust,
        )

        raw.to_pickle(price_cache_path)
        print(f"Saved price data to: {price_cache_path}")

    return raw, successful_tickers, failed_tickers