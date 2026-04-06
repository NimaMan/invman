#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path


ACCESS_DATE = "2026-04-06"
WINDOW_START = date.fromisoformat("2025-04-06")
WINDOW_END = date.fromisoformat("2026-04-06")
CRISIS_START = date.fromisoformat("2026-02-28")
PROBLEM_ROOT = Path(__file__).resolve().parents[2]
HISTORY_ROOT = PROBLEM_ROOT / "history"
RAW_DIR = HISTORY_ROOT / "data" / "raw"
PROCESSED_DIR = HISTORY_ROOT / "data" / "processed"
SOURCES_DIR = HISTORY_ROOT / "sources"
RESULTS_DIR = HISTORY_ROOT / "results"


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    title: str
    organization: str
    publication_date: str
    url: str
    local_path: str
    notes: str
    fetch_url: str | None = None


DOWNLOAD_SOURCES = [
    SourceSpec(
        source_id="fred_dcoilbrenteu_csv",
        title="Crude Oil Prices: Brent - Europe",
        organization="Federal Reserve Bank of St. Louis",
        publication_date="",
        url="https://fred.stlouisfed.org/series/DCOILBRENTEU",
        local_path="history/data/raw/fred_dcoilbrenteu.csv",
        notes="Live FRED CSV snapshot for the Brent daily series sourced from EIA.",
        fetch_url="https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU",
    ),
    SourceSpec(
        source_id="fred_dcoilwtico_csv",
        title="Crude Oil Prices: West Texas Intermediate (WTI) - Cushing, Oklahoma",
        organization="Federal Reserve Bank of St. Louis",
        publication_date="",
        url="https://fred.stlouisfed.org/series/DCOILWTICO",
        local_path="history/data/raw/fred_dcoilwtico.csv",
        notes="Live FRED CSV snapshot for the WTI daily series sourced from EIA.",
        fetch_url="https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILWTICO",
    ),
]


LOCAL_SOURCES = [
    SourceSpec(
        source_id="eia_daily_prices_2026_04_06_html",
        title="Today in Energy Daily Prices",
        organization="U.S. Energy Information Administration",
        publication_date="2026-04-06",
        url="https://www.eia.gov/todayinenergy/prices.php",
        local_path="data/raw/eia_daily_prices_2026-04-06.html",
        notes="Used to append the latest observed Brent and WTI close available on the analysis date.",
    ),
    SourceSpec(
        source_id="eia_steo_2026_03_pdf",
        title="Short-Term Energy Outlook, March 2026",
        organization="U.S. Energy Information Administration",
        publication_date="2026-03-10",
        url="https://www.eia.gov/outlooks/steo/pdf/steo_full.pdf",
        local_path="data/raw/eia_steo_full_2026-03.pdf",
        notes="Used for the March 2026 Brent expectation anchor under continued Hormuz disruption.",
    ),
    SourceSpec(
        source_id="opec_2026_04_05_production_decision_html",
        title="Saudi Arabia, Russia, Iraq, UAE, Kuwait, Kazakhstan, Algeria, and Oman adjust production and reaffirm commitment to market stability",
        organization="Organization of the Petroleum Exporting Countries",
        publication_date="2026-04-05",
        url="https://www.opec.org/pr-detail/597-5-april-2026.html?mod=livecoverage_web",
        local_path="data/raw/opec_2026-04-05_production_decision.html",
        notes="Used for the May 2026 206 kb/d supply adjustment anchor.",
    ),
    SourceSpec(
        source_id="eia_hormuz_tie_2025_06_16_html",
        title="Amid regional conflict, the Strait of Hormuz remains critical oil chokepoint",
        organization="U.S. Energy Information Administration",
        publication_date="2025-06-16",
        url="https://www.eia.gov/todayinenergy/detail.php?id=65504",
        local_path="data/raw/eia_hormuz_today_in_energy_2025-06-16.html",
        notes="Structural baseline source for Hormuz exposure and bypass capacity.",
    ),
    SourceSpec(
        source_id="mscio_jmic_update_003_2026_02_28_pdf",
        title="Update 003 - 001 - JMIC Advisory Note 28_FEB_2026_FINAL",
        organization="MSCIO / JMIC",
        publication_date="2026-03-04",
        url="https://www.mscio.eu/media/documents/Update_003_-_001_-_JMIC_Advisory_Note_28_FEB_2026_FINAL.pdf",
        local_path="maritime_traffic/data/raw/mscio_jmic_update_003_2026-02-28.pdf",
        notes="Conflict-start reference note for the Strait of Hormuz threat environment.",
    ),
    SourceSpec(
        source_id="mscio_jmic_update_006_2026_03_06_pdf",
        title="Update 006 JMIC Advisory Note 06_MAR_2026_FINAL",
        organization="MSCIO / JMIC",
        publication_date="2026-03-06",
        url="https://www.mscio.eu/media/documents/Update_006_JMIC_Advisory_Note_06_MAR_2026_FINAL.pdf",
        local_path="maritime_traffic/data/raw/mscio_jmic_update_006_2026-03-06.pdf",
        notes="Contains the sharp before-vs-after transit comparison spanning 28 February through 6 March 2026.",
    ),
    SourceSpec(
        source_id="mscio_jmic_update_008_2026_03_08_pdf",
        title="Update 008 - JMIC Advisory Note 08_MAR_2026",
        organization="MSCIO / JMIC",
        publication_date="2026-03-08",
        url="https://www.mscio.eu/media/documents/Update_008_-_JMIC_Advisory_Note_08_MAR_2026.pdf",
        local_path="maritime_traffic/data/raw/mscio_jmic_update_008_2026-03-08.pdf",
        notes="Contains AIS-derived cargo and tanker transits for 1 through 7 March 2026.",
    ),
    SourceSpec(
        source_id="mscio_jmic_update_010_2026_03_10_pdf",
        title="Update 010 - JMIC Advisory Note 10_MAR_2026_FINAL",
        organization="MSCIO / JMIC",
        publication_date="2026-03-13",
        url="https://www.mscio.eu/media/documents/Update_010_-_JMIC_Advisory_Note_10_MAR_2026_FINAL.pdf",
        local_path="maritime_traffic/data/raw/mscio_jmic_update_010_2026-03-10.pdf",
        notes="Contains AIS-derived cargo and tanker transits for 3 through 9 March 2026.",
    ),
    SourceSpec(
        source_id="mscio_jmic_update_016_2026_03_16_pdf",
        title="Update 016 - JMIC Advisory Note 16_MAR_2026_FINAL",
        organization="MSCIO / JMIC",
        publication_date="2026-03-20",
        url="https://www.mscio.eu/media/documents/Update_016_-_JMIC_Advisory_Note_16_MAR_2026_FINAL.pdf",
        local_path="maritime_traffic/data/raw/mscio_jmic_update_016_2026-03-16.pdf",
        notes="Contains 9 to 15 March 2026 Strait of Hormuz cargo and tanker transit tables and the ~138/day historical reference.",
    ),
    SourceSpec(
        source_id="mscio_jmic_update_018_2026_03_18_pdf",
        title="Update 018 - JMIC Advisory Note 18-Mar_FINAL",
        organization="MSCIO / JMIC",
        publication_date="2026-03-20",
        url="https://www.mscio.eu/media/documents/Update_018_-_JMIC_Advisory_Note_18-Mar_FINAL.pdf",
        local_path="maritime_traffic/data/raw/mscio_jmic_update_018_2026-03-18.pdf",
        notes="Contains 11 to 17 March 2026 Strait of Hormuz cargo and tanker transit tables.",
    ),
    SourceSpec(
        source_id="mscio_jmic_week_8_dashboard_2026_02_23_pdf",
        title="JMIC Week 8 Dashboard 16 Feb to 22 Feb 26",
        organization="MSCIO / JMIC",
        publication_date="2026-02-23",
        url="https://mscio.eu/media/documents/JMIC_Week_8_Dashboard_16_Feb_to_22_Feb_26.pdf",
        local_path="maritime_traffic/data/raw/mscio_jmic_week_8_dashboard_2026-02-23.pdf",
        notes="Pre-hostilities weekly baseline dashboard.",
    ),
    SourceSpec(
        source_id="unctad_rmt2025_ch2_pdf",
        title="Review of Maritime Transport 2025 Chapter II",
        organization="United Nations Conference on Trade and Development",
        publication_date="2025-10-22",
        url="https://unctad.org/system/files/official-document/rmt2025ch2_en.pdf",
        local_path="maritime_traffic/data/raw/unctad_review_of_maritime_transport_2025_ch2.pdf",
        notes="Contains the mid-June 2025 average ship transit reference for the Strait of Hormuz.",
    ),
]


CRISIS_TRAFFIC_ROWS = [
    ("2025-06-15", "pre_2026_war", "", "", 144.0, "unctad_rmt2025_ch2_pdf", "Mid-June 2025 average total ship transits."),
    ("2026-02-28", "pre_hostilities", 98.0, 50.0, 138.0, "mscio_jmic_update_006_2026_03_06_pdf", "Pre-hostilities AIS snapshot carried in the 6 March note."),
    ("2026-03-01", "acute_disruption", 18.0, 3.0, 138.0, "mscio_jmic_update_008_2026_03_08_pdf", "Day-one post-hostilities AIS snapshot."),
    ("2026-03-02", "acute_disruption", 7.0, 3.0, 138.0, "mscio_jmic_update_008_2026_03_08_pdf", "AIS-derived commercial traffic."),
    ("2026-03-03", "acute_disruption", 1.0, 0.0, 138.0, "mscio_jmic_update_010_2026_03_10_pdf", "AIS-derived commercial traffic."),
    ("2026-03-04", "acute_disruption", 2.0, 0.0, 138.0, "mscio_jmic_update_010_2026_03_10_pdf", "AIS-derived commercial traffic."),
    ("2026-03-05", "acute_disruption", 4.0, 2.0, 138.0, "mscio_jmic_update_010_2026_03_10_pdf", "AIS-derived commercial traffic."),
    ("2026-03-06", "acute_disruption", 4.0, 2.0, 138.0, "mscio_jmic_update_006_2026_03_06_pdf", "Near-total temporary pause still visible in the 6 March note."),
    ("2026-03-07", "acute_disruption", 1.0, 0.0, 138.0, "mscio_jmic_update_008_2026_03_08_pdf", "AIS-derived commercial traffic."),
    ("2026-03-08", "acute_disruption", 1.0, 0.0, 138.0, "mscio_jmic_update_010_2026_03_10_pdf", "AIS-derived commercial traffic."),
    ("2026-03-09", "acute_disruption", 1.0, 0.0, 138.0, "mscio_jmic_update_010_2026_03_10_pdf", "AIS-derived commercial traffic."),
    ("2026-03-10", "acute_disruption", 7.0, 1.0, 138.0, "mscio_jmic_update_016_2026_03_16_pdf", "Single-day rebound remained far below history."),
    ("2026-03-11", "prolonged_disruption", 1.0, 0.0, 138.0, "mscio_jmic_update_018_2026_03_18_pdf", "AIS-derived commercial traffic."),
    ("2026-03-12", "prolonged_disruption", 5.0, 0.0, 138.0, "mscio_jmic_update_018_2026_03_18_pdf", "AIS-derived commercial traffic."),
    ("2026-03-13", "prolonged_disruption", 1.0, 1.0, 138.0, "mscio_jmic_update_018_2026_03_18_pdf", "AIS-derived commercial traffic."),
    ("2026-03-14", "prolonged_disruption", 2.0, 0.0, 138.0, "mscio_jmic_update_018_2026_03_18_pdf", "AIS-derived commercial traffic."),
    ("2026-03-15", "prolonged_disruption", 3.0, 0.0, 138.0, "mscio_jmic_update_016_2026_03_16_pdf", "JMIC observed only three confirmed commercial cargo transits in the prior 24 hours."),
    ("2026-03-16", "prolonged_disruption", 2.0, 0.0, 138.0, "mscio_jmic_update_018_2026_03_18_pdf", "AIS-derived commercial traffic."),
    ("2026-03-17", "prolonged_disruption", 4.0, 0.0, 138.0, "mscio_jmic_update_018_2026_03_18_pdf", "JMIC observed four confirmed commercial vessel transits in the prior 24 hours."),
]


EVENT_TIMELINE_ROWS = [
    {
        "event_id": "hormuz_mid_june_2025_baseline",
        "date": "2025-06-15",
        "category": "shipping_baseline",
        "title": "Mid-June 2025 Strait of Hormuz baseline traffic remained normal",
        "impact_channel": "shipping_baseline",
        "direction": "neutral",
        "source_id": "unctad_rmt2025_ch2_pdf",
        "notes": "UNCTAD cites an average of 144 ship transits per day by mid-June 2025 and no significant change by end-June.",
    },
    {
        "event_id": "eia_hormuz_exposure_note",
        "date": "2025-06-16",
        "category": "structural_exposure",
        "title": "EIA quantifies Hormuz oil exposure and bypass limits",
        "impact_channel": "structural_exposure",
        "direction": "risk_up",
        "source_id": "eia_hormuz_tie_2025_06_16_html",
        "notes": "EIA states 2024 Hormuz oil flow near 20 mb/d and available bypass capacity around 2.6 mb/d.",
    },
    {
        "event_id": "hostilities_begin",
        "date": "2026-02-28",
        "category": "conflict_start",
        "title": "Regional hostilities begin",
        "impact_channel": "geopolitical_risk",
        "direction": "risk_up",
        "source_id": "mscio_jmic_update_003_2026_02_28_pdf",
        "notes": "JMIC uses 28 February 2026 as the start of the current escalation and reports reduced but ongoing Strait traffic with no formal closure.",
    },
    {
        "event_id": "traffic_collapse_visible",
        "date": "2026-03-06",
        "category": "shipping_disruption",
        "title": "AIS-derived Hormuz traffic collapses after the conflict start",
        "impact_channel": "transit_capacity",
        "direction": "risk_up",
        "source_id": "mscio_jmic_update_006_2026_03_06_pdf",
        "notes": "JMIC shows cargo traffic falling from 98 on 28 February to 4 on 6 March and tanker traffic from 50 to 2.",
    },
    {
        "event_id": "eia_steo_march_anchor",
        "date": "2026-03-10",
        "category": "market_expectation",
        "title": "EIA STEO keeps Brent above $95 over the next two months",
        "impact_channel": "price_expectation",
        "direction": "risk_up",
        "source_id": "eia_steo_2026_03_pdf",
        "notes": "The March 2026 STEO says Brent stays above $95/b over the next two months and averages $91/b in 2Q26 because of reduced Hormuz flows.",
    },
    {
        "event_id": "persistent_mid_march_disruption",
        "date": "2026-03-16",
        "category": "shipping_disruption",
        "title": "Hormuz cargo traffic remains far below historical averages in mid-March",
        "impact_channel": "transit_capacity",
        "direction": "risk_up",
        "source_id": "mscio_jmic_update_016_2026_03_16_pdf",
        "notes": "JMIC historical reference is about 138 vessels/day while only three cargo transits were confirmed in the prior 24 hours.",
    },
    {
        "event_id": "critical_risk_continues",
        "date": "2026-03-18",
        "category": "shipping_disruption",
        "title": "Critical maritime risk persists despite no new confirmed attacks since 12 March",
        "impact_channel": "geopolitical_risk",
        "direction": "risk_up",
        "source_id": "mscio_jmic_update_018_2026_03_18_pdf",
        "notes": "JMIC still reports only four confirmed commercial vessel transits in the prior 24 hours and maintains a critical risk level.",
    },
    {
        "event_id": "opec_may_adjustment",
        "date": "2026-04-05",
        "category": "supply_response",
        "title": "OPEC+ announces a 206 kb/d May 2026 production adjustment",
        "impact_channel": "supply_response",
        "direction": "risk_down",
        "source_id": "opec_2026_04_05_production_decision_html",
        "notes": "This is the explicit non-Hormuz supply response anchor used in the first scenario engine.",
    },
    {
        "event_id": "latest_price_anchor",
        "date": "2026-04-06",
        "category": "price_anchor",
        "title": "EIA daily prices page publishes the latest Brent and WTI close used in the analysis",
        "impact_channel": "price_observation",
        "direction": "neutral",
        "source_id": "eia_daily_prices_2026_04_06_html",
        "notes": "The page dated 6 April 2026 reports Brent at 127.61 and WTI at 113.23 for the latest observed close on 2 April 2026.",
    },
]


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def download(url: str) -> bytes:
    try:
        result = subprocess.run(
            [
                "curl",
                "--http1.1",
                "-L",
                "--retry",
                "3",
                "--retry-delay",
                "1",
                "--connect-timeout",
                "20",
                "--max-time",
                "60",
                "-A",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
                url,
            ],
            check=True,
            capture_output=True,
            timeout=90,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Failed to download {url}: {stderr}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Timed out while downloading {url}") from exc
    return result.stdout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the reproducible one-year Hormuz backtest dataset."
    )
    parser.add_argument(
        "--refresh-downloads",
        action="store_true",
        help="Refresh network-downloadable raw files before rebuilding. By default the script reuses checked-in raw snapshots.",
    )
    return parser.parse_args()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_market_anchors() -> dict[str, str]:
    path = PROBLEM_ROOT / "data" / "processed" / "current_market_anchors.csv"
    anchors: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            anchors[row["anchor_id"]] = row["value"]
    return anchors


def read_series(path: Path, value_key: str) -> dict[date, float]:
    values: dict[date, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw_value = row[value_key].strip()
            if not raw_value or raw_value == ".":
                continue
            observation_date = date.fromisoformat(row["observation_date"])
            if WINDOW_START <= observation_date <= WINDOW_END:
                values[observation_date] = float(raw_value)
    return values


def build_daily_prices() -> list[dict[str, str]]:
    brent = read_series(RAW_DIR / "fred_dcoilbrenteu.csv", "DCOILBRENTEU")
    wti = read_series(RAW_DIR / "fred_dcoilwtico.csv", "DCOILWTICO")
    anchors = read_market_anchors()
    latest_close_date = date.fromisoformat(anchors["latest_observed_close_date"])
    brent[latest_close_date] = float(anchors["latest_observed_brent_usd_per_bbl"])
    wti[latest_close_date] = float(anchors["latest_observed_wti_usd_per_bbl"])

    rows: list[dict[str, str]] = []
    for observation_date in sorted(set(brent) | set(wti)):
        row = {
            "date": observation_date.isoformat(),
            "brent_usd_per_bbl": f"{brent.get(observation_date, float('nan')):.2f}"
            if observation_date in brent
            else "",
            "wti_usd_per_bbl": f"{wti.get(observation_date, float('nan')):.2f}"
            if observation_date in wti
            else "",
            "brent_source_id": "eia_daily_prices_2026_04_06_html"
            if observation_date == latest_close_date
            else "fred_dcoilbrenteu_csv",
            "wti_source_id": "eia_daily_prices_2026_04_06_html"
            if observation_date == latest_close_date
            else "fred_dcoilwtico_csv",
            "notes": "EIA daily-prices anchor appended to FRED history snapshot"
            if observation_date == latest_close_date
            else "",
        }
        rows.append(row)
    return rows


def build_monthly_prices(daily_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in daily_rows:
        grouped[row["date"][:7]].append(row)

    monthly_rows: list[dict[str, str]] = []
    for year_month in sorted(grouped):
        rows = grouped[year_month]
        brent_values = [float(row["brent_usd_per_bbl"]) for row in rows if row["brent_usd_per_bbl"]]
        wti_values = [float(row["wti_usd_per_bbl"]) for row in rows if row["wti_usd_per_bbl"]]
        monthly_rows.append(
            {
                "year_month": year_month,
                "window_start": rows[0]["date"],
                "window_end": rows[-1]["date"],
                "observation_count": str(len(rows)),
                "brent_avg_usd_per_bbl": f"{sum(brent_values) / len(brent_values):.2f}"
                if brent_values
                else "",
                "brent_min_usd_per_bbl": f"{min(brent_values):.2f}" if brent_values else "",
                "brent_max_usd_per_bbl": f"{max(brent_values):.2f}" if brent_values else "",
                "wti_avg_usd_per_bbl": f"{sum(wti_values) / len(wti_values):.2f}"
                if wti_values
                else "",
                "wti_min_usd_per_bbl": f"{min(wti_values):.2f}" if wti_values else "",
                "wti_max_usd_per_bbl": f"{max(wti_values):.2f}" if wti_values else "",
            }
        )
    return monthly_rows


def build_shipping_proxies() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for observation_date, phase, cargo, tanker, historical, source_id, notes in CRISIS_TRAFFIC_ROWS:
        rows.append(
            {
                "date": observation_date,
                "phase": phase,
                "observed_cargo_transits_ais": f"{cargo:.0f}" if cargo != "" else "",
                "observed_tanker_transits_ais": f"{tanker:.0f}" if tanker != "" else "",
                "historical_average_total_ship_transits": f"{historical:.0f}" if historical else "",
                "source_id": source_id,
                "notes": notes,
            }
        )
    return rows


def ensure_source_present(spec: SourceSpec, refresh_downloads: bool) -> None:
    local_path = PROBLEM_ROOT / spec.local_path
    if spec.fetch_url is not None and (refresh_downloads or not local_path.exists()):
        try:
            payload = download(spec.fetch_url or spec.url)
        except RuntimeError:
            if local_path.exists():
                print(
                    f"Warning: could not refresh {spec.source_id}; reusing existing local snapshot at {local_path}"
                )
                return
            raise
        local_path.write_bytes(payload)
        return
    if not local_path.exists():
        raise FileNotFoundError(
            f"Missing required local source file: {local_path}. "
            "Re-run with --refresh-downloads if this source is downloadable."
        )


def write_sources(refresh_downloads: bool) -> None:
    manifest_rows: list[dict[str, str]] = []
    checksum_lines: list[str] = []

    for spec in DOWNLOAD_SOURCES:
        ensure_source_present(spec, refresh_downloads)

    for spec in DOWNLOAD_SOURCES + LOCAL_SOURCES:
        ensure_source_present(spec, refresh_downloads=False)
        local_path = PROBLEM_ROOT / spec.local_path
        digest = sha256_file(local_path)
        manifest_rows.append(
            {
                "source_id": spec.source_id,
                "title": spec.title,
                "organization": spec.organization,
                "publication_date": spec.publication_date,
                "access_date": ACCESS_DATE,
                "url": spec.url,
                "local_path": spec.local_path,
                "sha256": digest,
                "notes": spec.notes,
            }
        )
        checksum_lines.append(f"{digest}  {spec.local_path}")

    write_csv(
        SOURCES_DIR / "source_manifest.csv",
        [
            "source_id",
            "title",
            "organization",
            "publication_date",
            "access_date",
            "url",
            "local_path",
            "sha256",
            "notes",
        ],
        manifest_rows,
    )
    (SOURCES_DIR / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def pct_change(current: float, baseline: float) -> float | None:
    if baseline == 0:
        return None
    return (current - baseline) / baseline * 100.0


def build_summary_payload(
    daily_prices: list[dict[str, str]], shipping_proxies: list[dict[str, str]]
) -> dict[str, object]:
    brent_series = [
        (date.fromisoformat(row["date"]), float(row["brent_usd_per_bbl"]))
        for row in daily_prices
        if row["brent_usd_per_bbl"]
    ]
    wti_series = [
        (date.fromisoformat(row["date"]), float(row["wti_usd_per_bbl"]))
        for row in daily_prices
        if row["wti_usd_per_bbl"]
    ]
    crisis_brent = [(day, value) for day, value in brent_series if day >= CRISIS_START]
    pre_crisis_window_start = CRISIS_START - timedelta(days=30)
    pre_crisis_brent = [
        value for day, value in brent_series if pre_crisis_window_start <= day < CRISIS_START
    ]
    crisis_brent_values = [value for _, value in crisis_brent]
    crisis_peak_day, crisis_peak_value = max(crisis_brent, key=lambda item: item[1])

    shipping_totals: list[tuple[date, float]] = []
    for row in shipping_proxies:
        cargo = row["observed_cargo_transits_ais"]
        tanker = row["observed_tanker_transits_ais"]
        if not cargo and not tanker:
            continue
        total = float(cargo or 0.0) + float(tanker or 0.0)
        shipping_totals.append((date.fromisoformat(row["date"]), total))

    pre_hostilities_total = next(
        total for day, total in shipping_totals if day == CRISIS_START
    )
    trough_day, trough_total = min(
        (item for item in shipping_totals if item[0] >= CRISIS_START),
        key=lambda item: item[1],
    )
    latest_shipping_day, latest_shipping_total = max(shipping_totals, key=lambda item: item[0])

    payload = {
        "dataset_name": "one_year_hormuz_backtest",
        "access_date": ACCESS_DATE,
        "window_start": WINDOW_START.isoformat(),
        "window_end": WINDOW_END.isoformat(),
        "crisis_start": CRISIS_START.isoformat(),
        "brent": {
            "first_observation_date": brent_series[0][0].isoformat(),
            "first_observation_usd_per_bbl": brent_series[0][1],
            "latest_observation_date": brent_series[-1][0].isoformat(),
            "latest_observation_usd_per_bbl": brent_series[-1][1],
            "window_average_usd_per_bbl": mean([value for _, value in brent_series]),
            "window_min_usd_per_bbl": min(value for _, value in brent_series),
            "window_max_usd_per_bbl": max(value for _, value in brent_series),
            "change_from_window_start_pct": pct_change(brent_series[-1][1], brent_series[0][1]),
            "pre_crisis_30d_average_usd_per_bbl": mean(pre_crisis_brent),
            "crisis_window_average_usd_per_bbl": mean(crisis_brent_values),
            "crisis_peak_date": crisis_peak_day.isoformat(),
            "crisis_peak_usd_per_bbl": crisis_peak_value,
        },
        "wti": {
            "first_observation_date": wti_series[0][0].isoformat(),
            "first_observation_usd_per_bbl": wti_series[0][1],
            "latest_observation_date": wti_series[-1][0].isoformat(),
            "latest_observation_usd_per_bbl": wti_series[-1][1],
            "window_average_usd_per_bbl": mean([value for _, value in wti_series]),
            "window_min_usd_per_bbl": min(value for _, value in wti_series),
            "window_max_usd_per_bbl": max(value for _, value in wti_series),
            "change_from_window_start_pct": pct_change(wti_series[-1][1], wti_series[0][1]),
        },
        "shipping": {
            "pre_hostilities_date": CRISIS_START.isoformat(),
            "pre_hostilities_observed_commercial_transits": pre_hostilities_total,
            "trough_date": trough_day.isoformat(),
            "trough_observed_commercial_transits": trough_total,
            "trough_drop_vs_pre_hostilities_pct": pct_change(trough_total, pre_hostilities_total),
            "latest_observation_date": latest_shipping_day.isoformat(),
            "latest_observed_commercial_transits": latest_shipping_total,
            "latest_drop_vs_pre_hostilities_pct": pct_change(
                latest_shipping_total, pre_hostilities_total
            ),
        },
        "interpretation": {
            "price_series_note": "FRED daily spot series carry the year-long history, with the latest EIA daily-prices close appended on 2026-04-02.",
            "shipping_series_note": "Shipping counts are AIS-derived lower-bound observations during the crisis window and indicate disruption severity rather than exact physical throughput.",
        },
    }
    return payload


def write_summary_results(
    daily_prices: list[dict[str, str]], shipping_proxies: list[dict[str, str]]
) -> None:
    payload = build_summary_payload(daily_prices, shipping_proxies)
    brent = payload["brent"]
    wti = payload["wti"]
    shipping = payload["shipping"]
    markdown = "\n".join(
        [
            "# one_year_backtest_summary",
            "",
            "Generated from the reproducible one-year Hormuz history package.",
            "",
            "## Key results",
            "",
            f"- Brent moved from `${brent['first_observation_usd_per_bbl']:.2f}/b` on `{brent['first_observation_date']}` to `${brent['latest_observation_usd_per_bbl']:.2f}/b` on `{brent['latest_observation_date']}`, a `{brent['change_from_window_start_pct']:.1f}%` increase over the backtest window.",
            f"- WTI moved from `${wti['first_observation_usd_per_bbl']:.2f}/b` on `{wti['first_observation_date']}` to `${wti['latest_observation_usd_per_bbl']:.2f}/b` on `{wti['latest_observation_date']}`, a `{wti['change_from_window_start_pct']:.1f}%` increase.",
            f"- Brent averaged `${brent['window_average_usd_per_bbl']:.2f}/b` across the full year, with a pre-crisis 30-day average of `${brent['pre_crisis_30d_average_usd_per_bbl']:.2f}/b` and a crisis-window average of `${brent['crisis_window_average_usd_per_bbl']:.2f}/b`.",
            f"- The highest Brent observation in the crisis window was `${brent['crisis_peak_usd_per_bbl']:.2f}/b` on `{brent['crisis_peak_date']}`.",
            f"- Observed AIS commercial transits through Hormuz fell from `{shipping['pre_hostilities_observed_commercial_transits']:.0f}` on `{shipping['pre_hostilities_date']}` to a trough of `{shipping['trough_observed_commercial_transits']:.0f}` on `{shipping['trough_date']}`, a `{shipping['trough_drop_vs_pre_hostilities_pct']:.1f}%` collapse.",
            f"- By `{shipping['latest_observation_date']}`, observed AIS commercial transits were still only `{shipping['latest_observed_commercial_transits']:.0f}`, a `{shipping['latest_drop_vs_pre_hostilities_pct']:.1f}%` gap versus the pre-hostilities snapshot.",
            "",
            "## Interpretation",
            "",
            "- The backtest window contains a clear regime break after `2026-02-28`: prices rise sharply while observed commercial transits collapse to single digits.",
            "- This is enough evidence for a first-degree FlowNet to treat Hormuz transit capacity as a time-varying shock state rather than a static parameter.",
            "- Shipping counts should be interpreted as lower-bound disruption indicators because JMIC repeatedly warns about AIS suppression, GNSS disruption, and possible dark transits.",
            "",
            "## Rebuild",
            "",
            "```bash",
            "python rust/src/problems/hormuz_strait/history/scripts/fetch_and_build.py",
            "```",
            "",
            "Use `--refresh-downloads` only when intentionally refreshing the downloadable raw snapshots.",
        ]
    )
    (RESULTS_DIR / "one_year_backtest_summary.md").write_text(markdown + "\n", encoding="utf-8")
    (RESULTS_DIR / "one_year_backtest_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    ensure_dirs()
    write_sources(refresh_downloads=args.refresh_downloads)

    daily_prices = build_daily_prices()
    monthly_prices = build_monthly_prices(daily_prices)
    shipping_proxies = build_shipping_proxies()

    write_csv(
        PROCESSED_DIR / "brent_wti_daily_prices.csv",
        [
            "date",
            "brent_usd_per_bbl",
            "wti_usd_per_bbl",
            "brent_source_id",
            "wti_source_id",
            "notes",
        ],
        daily_prices,
    )
    write_csv(
        PROCESSED_DIR / "brent_wti_monthly_summary.csv",
        [
            "year_month",
            "window_start",
            "window_end",
            "observation_count",
            "brent_avg_usd_per_bbl",
            "brent_min_usd_per_bbl",
            "brent_max_usd_per_bbl",
            "wti_avg_usd_per_bbl",
            "wti_min_usd_per_bbl",
            "wti_max_usd_per_bbl",
        ],
        monthly_prices,
    )
    write_csv(
        PROCESSED_DIR / "hormuz_market_event_timeline.csv",
        [
            "event_id",
            "date",
            "category",
            "title",
            "impact_channel",
            "direction",
            "source_id",
            "notes",
        ],
        EVENT_TIMELINE_ROWS,
    )
    write_csv(
        PROCESSED_DIR / "hormuz_shipping_disruption_daily_signals.csv",
        [
            "date",
            "phase",
            "observed_cargo_transits_ais",
            "observed_tanker_transits_ais",
            "historical_average_total_ship_transits",
            "source_id",
            "notes",
        ],
        shipping_proxies,
    )
    write_summary_results(daily_prices, shipping_proxies)

    print("Built Hormuz history backtest dataset under:")
    print(HISTORY_ROOT)


if __name__ == "__main__":
    main()
