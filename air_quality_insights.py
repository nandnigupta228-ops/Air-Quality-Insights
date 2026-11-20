"""Run with: python air_quality_insights.py [path/to/04_air_quality.csv]"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd


# Project analysis threshold, in the same units as PM2.5 in the CSV.
# This is a descriptive cutoff, not an official air-quality standard.
THRESHOLD = 60


def show(title, table):
    print(f"\n{title}")
    print(table.round(2).to_string())


def category_column(pm):
    """Read instructor-defined ranges. Upper boundaries are inclusive."""
    print("\n57. Enter the instructor's category rules, lowest range first.")
    print("Use inclusive upper limits. Leave the first prompt blank to skip.")
    while True:
        raw = input("Upper limits, comma-separated (exclude infinity): ").strip()
        if not raw:
            return pd.Series("Pending instructor thresholds", index=pm.index)
        try:
            limits = [float(item) for item in raw.split(",")]
            labels = [item.strip() for item in input(
                "Category names, lowest to highest (include the final range): "
            ).split(",")]
            if not all(np.isfinite(limits)) or limits != sorted(set(limits)):
                raise ValueError("Use finite, strictly increasing upper limits.")
            if len(labels) != len(limits) + 1 or not all(labels) or len(set(labels)) != len(labels):
                raise ValueError("Enter one distinct name per range, including the last.")
            return pd.cut(pm, [-np.inf, *limits, np.inf], labels=labels)
        except ValueError as error:
            print(f"Please try again. {error}")


def analyze(path):
    # 49-51. Inspect the original data before handling missing readings.
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()
    required = {"Date", "City", "PM2.5", "PM10", "Temperature"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError("Missing CSV columns: " + ", ".join(sorted(missing)))
    print(f"49. Dataset: {len(data)} rows, {len(data.columns)} columns")
    data.info()
    show("First five rows", data.head())
    data = data.replace(r"^\s*$", np.nan, regex=True)
    show("50. Missing values per input column", data.isna().sum())
    for column in ["PM2.5", "PM10", "Temperature"]:
        data[column] = pd.to_numeric(data[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )

    # Available-case analysis keeps missing readings from becoming invented peaks.
    # The original rows remain available for other measurements.
    pm = data["PM2.5"].dropna()
    if pm.empty:
        raise ValueError("No valid PM2.5 readings are available.")
    print(f"\n51. Using {len(pm)} observed PM2.5 readings. Excluded "
          f"{len(data) - len(pm)} missing or invalid readings from PM2.5 calculations. "
          "This avoids estimating exposure on days without measurements.")

    data["City"] = data["City"].astype("string").str.strip()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Month"] = data["Date"].dt.to_period("M")
    show("52. Extracted year and month", data[["Date", "Month"]].head())
    show("53. Overall PM2.5 statistics", pm.agg(["mean", "median", "max"]))

    cities = data.groupby("City")["PM2.5"].agg(["count", "mean"]).dropna()
    cities = cities.sort_values("mean", ascending=False)
    cities["rank"] = cities["mean"].rank(ascending=False, method="min").astype(int)
    show("54. City ranking by mean PM2.5", cities[["rank", "mean", "count"]])
    peaks = data.loc[data["PM2.5"].eq(pm.max()), ["Date", "City", "PM2.5"]]
    show("55. Highest observed PM2.5 (including ties)", peaks)
    pairs = data.dropna(subset=["PM10", "PM2.5"])
    show("56. City means on rows with both pollutants",
         pairs.groupby("City")[["PM10", "PM2.5"]].mean())

    data["Air_Quality_Category"] = category_column(data["PM2.5"])
    data.loc[data["PM2.5"].isna(), "Air_Quality_Category"] = np.nan
    show("57. Air-quality category counts (NaN = missing reading)",
         data["Air_Quality_Category"].value_counts(dropna=False))
    monthly = data.groupby(["City", "Month"])["PM2.5"].agg(["mean", "count"])
    show("58. Monthly PM2.5 by city", monthly)

    print(f"\n59. Population standard deviation: {np.std(pm):.2f}")
    levels = [10, 25, 50, 75, 90]
    show("PM2.5 percentiles", pd.Series(np.percentile(pm, levels), index=levels))
    q1, q3 = np.percentile(pm, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    unusual = data["PM2.5"].lt(lower) | data["PM2.5"].gt(upper)
    print(f"\n60. IQR outliers: {unusual.sum()} (fences {lower:.2f}, {upper:.2f})")
    if unusual.any():
        show("Outlier list", data.loc[unusual, ["Date", "City", "PM2.5"]])

    paired = data[["Temperature", "PM2.5"]].dropna()
    if len(paired) >= 2 and paired.nunique().min() > 1:
        r = paired["Temperature"].corr(paired["PM2.5"])
        print(f"\n61. Temperature vs PM2.5: Pearson r = {r:.3f} ({len(paired)} pairs)")
    else:
        print("\n61. Correlation undefined: insufficient paired data or variation.")

    above = int((pm > THRESHOLD).sum())
    print(f"\n62. PM2.5 above project cutoff {THRESHOLD}: "
          f"{above}/{len(pm)} = {above / len(pm):.1%}")
    print("\n63. Five data-supported public-awareness recommendations")
    if not cities.empty:
        print(f"1. Prioritize awareness sessions in {cities.index[0]}, where "
              f"average PM2.5 reached {cities.iloc[0]['mean']:.1f}.")
    else:
        print("1. Request city labels before publishing location-specific guidance.")
    if monthly["mean"].notna().any():
        city, month = monthly["mean"].idxmax()
        print(f"2. Highlight {city} in {month} in the public bulletin: its "
              f"monthly average of {monthly['mean'].max():.1f} was the highest observed.")
    else:
        print("2. Request dated, city-labelled readings before issuing monthly bulletins.")
    peak = peaks.iloc[0]
    print(f"3. Use the peak reading of {pm.max():.1f} in a public example "
          f"of how to read local monitoring reports ({peak['City']}, {peak['Date']:%Y-%m-%d})."
          if pd.notna(peak["Date"]) else
          f"3. Report the peak of {pm.max():.1f} with its date marked unavailable.")
    print(f"4. Include the frequency of elevated readings in updates: {above} of "
          f"{len(pm)} exceeded the project's comparison level of {THRESHOLD}.")
    print(f"5. Publish monitoring coverage alongside averages: {len(pm)} of "
          f"{len(data)} records had usable PM2.5 readings.")


if __name__ == "__main__":
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("04_air_quality.csv")
    try:
        analyze(csv_path)
    except (OSError, ValueError) as error:
        sys.exit(f"Could not analyze the CSV: {error}")
