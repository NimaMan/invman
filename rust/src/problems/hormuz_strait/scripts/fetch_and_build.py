#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import io
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path


ACCESS_DATE = "2026-04-06"
PROBLEM_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROBLEM_ROOT / "data" / "raw"
PROCESSED_DIR = PROBLEM_ROOT / "data" / "processed"
SOURCES_DIR = PROBLEM_ROOT / "sources"


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    title: str
    organization: str
    publication_date: str
    url: str
    raw_filename: str
    notes: str


SOURCES = [
    SourceSpec(
        source_id="eia_hormuz_tie_2025_06_16_html",
        title="Amid regional conflict, the Strait of Hormuz remains critical oil chokepoint",
        organization="U.S. Energy Information Administration",
        publication_date="2025-06-16",
        url="https://www.eia.gov/todayinenergy/detail.php?id=65504",
        raw_filename="eia_hormuz_today_in_energy_2025-06-16.html",
        notes="Primary narrative source for 2024 Hormuz flow, bypass, and destination statements.",
    ),
    SourceSpec(
        source_id="eia_hormuz_fig1_2025_06_16_xlsx",
        title="EIA figure data: total oil flows through the Strait of Hormuz",
        organization="U.S. Energy Information Administration",
        publication_date="2025-06-16",
        url="https://www.eia.gov/todayinenergy/images/2025.06.16/fig1.xlsx",
        raw_filename="eia_hormuz_fig1_2025-06-16.xlsx",
        notes="Contains annual total flow series for 2020-1Q25.",
    ),
    SourceSpec(
        source_id="eia_hormuz_fig3_2025_06_16_xlsx",
        title="EIA figure data: Hormuz origin and destination flows",
        organization="U.S. Energy Information Administration",
        publication_date="2025-06-16",
        url="https://www.eia.gov/todayinenergy/images/2025.06.16/fig3.xlsx",
        raw_filename="eia_hormuz_fig3_2025-06-16.xlsx",
        notes="Contains 2020-1Q25 origin-country and destination-market series.",
    ),
    SourceSpec(
        source_id="eia_top_oil_producers_consumers_faq_2024_04_11_html",
        title="What countries are the top producers and consumers of oil?",
        organization="U.S. Energy Information Administration",
        publication_date="2024-04-11",
        url="https://www.eia.gov/tools/faqs/faq.php?id=709&t=6",
        raw_filename="eia_top_producers_consumers_faq_2024-04-11.html",
        notes="Used to justify geography-first node selection over company-level nodes.",
    ),
    SourceSpec(
        source_id="eia_daily_prices_2026_04_06_html",
        title="Today in Energy Daily Prices",
        organization="U.S. Energy Information Administration",
        publication_date="2026-04-06",
        url="https://www.eia.gov/todayinenergy/prices.php",
        raw_filename="eia_daily_prices_2026-04-06.html",
        notes="Used for the latest observed Brent and WTI close available on the analysis date.",
    ),
    SourceSpec(
        source_id="eia_steo_2026_03_pdf",
        title="Short-Term Energy Outlook, March 2026",
        organization="U.S. Energy Information Administration",
        publication_date="2026-03-10",
        url="https://www.eia.gov/outlooks/steo/pdf/steo_full.pdf",
        raw_filename="eia_steo_full_2026-03.pdf",
        notes="Used for the Brent floor and 2Q26 average anchor under continuing Hormuz disruption.",
    ),
    SourceSpec(
        source_id="opec_2026_04_05_production_decision_html",
        title="Saudi Arabia, Russia, Iraq, UAE, Kuwait, Kazakhstan, Algeria, and Oman adjust production and reaffirm commitment to market stability",
        organization="Organization of the Petroleum Exporting Countries",
        publication_date="2026-04-05",
        url="https://www.opec.org/pr-detail/597-5-april-2026.html?mod=livecoverage_web",
        raw_filename="opec_2026-04-05_production_decision.html",
        notes="Used for the May 2026 206 kb/d supply adjustment anchor and the statement on alternative export routes.",
    ),
]


NODE_SET_V1 = [
    {
        "node_id": "saudi_arabia_origin",
        "label": "Saudi Arabia origin exports",
        "role": "origin_exporter",
        "baseline_flow_million_bpd_2024": "5.477689579234973",
        "selection_basis": "eia_hormuz_fig3_origin_2024",
        "notes": "Largest origin flow in the 2024 EIA figure data.",
    },
    {
        "node_id": "iraq_origin",
        "label": "Iraq origin exports",
        "role": "origin_exporter",
        "baseline_flow_million_bpd_2024": "3.223645852459016",
        "selection_basis": "eia_hormuz_fig3_origin_2024",
        "notes": "Second-largest origin flow in the 2024 EIA figure data.",
    },
    {
        "node_id": "uae_origin",
        "label": "United Arab Emirates origin exports",
        "role": "origin_exporter",
        "baseline_flow_million_bpd_2024": "1.8900929781420766",
        "selection_basis": "eia_hormuz_fig3_origin_2024",
        "notes": "Major exporter with bypass optionality outside Hormuz.",
    },
    {
        "node_id": "iran_origin",
        "label": "Iran origin exports",
        "role": "origin_exporter",
        "baseline_flow_million_bpd_2024": "1.3996683715846996",
        "selection_basis": "eia_hormuz_fig3_origin_2024",
        "notes": "Material Hormuz exporter in the current pattern.",
    },
    {
        "node_id": "kuwait_origin",
        "label": "Kuwait origin exports",
        "role": "origin_exporter",
        "baseline_flow_million_bpd_2024": "1.3257541338797816",
        "selection_basis": "eia_hormuz_fig3_origin_2024",
        "notes": "Material Hormuz exporter in the current pattern.",
    },
    {
        "node_id": "qatar_origin",
        "label": "Qatar origin exports",
        "role": "origin_exporter",
        "baseline_flow_million_bpd_2024": "0.6490193579234973",
        "selection_basis": "eia_hormuz_fig3_origin_2024",
        "notes": "Oil and condensate only; LNG remains out of scope in v1.",
    },
    {
        "node_id": "other_hormuz_origins",
        "label": "Other Hormuz exporters",
        "role": "origin_exporter",
        "baseline_flow_million_bpd_2024": "0.35274353551912263",
        "selection_basis": "eia_hormuz_fig3_origin_2024",
        "notes": "Residual exporter bucket.",
    },
    {
        "node_id": "strait_of_hormuz",
        "label": "Strait of Hormuz chokepoint",
        "role": "transit_asset",
        "baseline_flow_million_bpd_2024": "20.261741721311477",
        "selection_basis": "eia_hormuz_fig1_2024",
        "notes": "Primary disrupted maritime chokepoint.",
    },
    {
        "node_id": "aggregate_bypass_capacity",
        "label": "Aggregate Saudi and UAE bypass capacity",
        "role": "transit_asset",
        "baseline_flow_million_bpd_2024": "2.6",
        "selection_basis": "eia_hormuz_tie_2025_alt_routes",
        "notes": "EIA estimate of effective unused bypass capacity.",
    },
    {
        "node_id": "china_market",
        "label": "China destination market",
        "role": "demand_market",
        "baseline_flow_million_bpd_2024": "4.7846856393442625",
        "selection_basis": "eia_hormuz_fig3_destination_2024",
        "notes": "Largest destination market in the 2024 EIA figure data.",
    },
    {
        "node_id": "india_market",
        "label": "India destination market",
        "role": "demand_market",
        "baseline_flow_million_bpd_2024": "1.8853302295081966",
        "selection_basis": "eia_hormuz_fig3_destination_2024",
        "notes": "Top Asian destination market.",
    },
    {
        "node_id": "south_korea_market",
        "label": "South Korea destination market",
        "role": "demand_market",
        "baseline_flow_million_bpd_2024": "1.7266257540983607",
        "selection_basis": "eia_hormuz_fig3_destination_2024",
        "notes": "Top Asian destination market.",
    },
    {
        "node_id": "japan_market",
        "label": "Japan destination market",
        "role": "demand_market",
        "baseline_flow_million_bpd_2024": "1.5153605710382514",
        "selection_basis": "eia_hormuz_fig3_destination_2024",
        "notes": "Top Asian destination market.",
    },
    {
        "node_id": "other_asia_market",
        "label": "Other Asia destination market",
        "role": "demand_market",
        "baseline_flow_million_bpd_2024": "2.071853393442623",
        "selection_basis": "eia_hormuz_fig3_destination_2024",
        "notes": "Residual Asian destination bucket.",
    },
    {
        "node_id": "europe_market",
        "label": "Europe destination market",
        "role": "demand_market",
        "baseline_flow_million_bpd_2024": "0.7206034344262295",
        "selection_basis": "eia_hormuz_fig3_destination_2024",
        "notes": "European destination bucket.",
    },
    {
        "node_id": "united_states_market",
        "label": "United States destination market",
        "role": "demand_market",
        "baseline_flow_million_bpd_2024": "0.4843527459016393",
        "selection_basis": "eia_hormuz_fig3_destination_2024",
        "notes": "Smaller direct receiver with larger indirect price exposure.",
    },
    {
        "node_id": "saudi_arabia_market",
        "label": "Saudi Arabia destination market",
        "role": "demand_market",
        "baseline_flow_million_bpd_2024": "0.22803622404371585",
        "selection_basis": "eia_hormuz_fig3_destination_2024",
        "notes": "Represents local Gulf demand visible in the destination data.",
    },
    {
        "node_id": "other_destinations_market",
        "label": "Other destination markets",
        "role": "demand_market",
        "baseline_flow_million_bpd_2024": "0.9017658169398892",
        "selection_basis": "eia_hormuz_fig3_destination_2024",
        "notes": "Residual destination bucket.",
    },
    {
        "node_id": "gulf_refining_and_storage_hub",
        "label": "Gulf refining and storage hub",
        "role": "refining_storage_hub",
        "baseline_flow_million_bpd_2024": "0.0",
        "selection_basis": "eia_hormuz_tie_2025_local_refining_note",
        "notes": "Captures the documented rise in local Gulf refining and storage absorption.",
    },
    {
        "node_id": "strategic_reserve_and_floating_storage",
        "label": "Strategic reserve and floating storage",
        "role": "reserve_buffer",
        "baseline_flow_million_bpd_2024": "0.0",
        "selection_basis": "modeling_assumption_v1",
        "notes": "Inventory-response buffer added so reserve release is explicit in the model.",
    },
]


SCENARIO_PARAMETERS_V1 = [
    {
        "parameter": "baseline_year",
        "value": "2024",
        "units": "year",
        "source_id": "eia_hormuz_tie_2025_06_16_html",
        "notes": "The initial model is anchored to 2024 flow patterns.",
    },
    {
        "parameter": "node_count",
        "value": "20",
        "units": "count",
        "source_id": "modeling_assumption_v1",
        "notes": "Mixed exporter, transit, market, and reserve node set.",
    },
    {
        "parameter": "closure_fraction",
        "value": "1.0",
        "units": "share",
        "source_id": "modeling_assumption_v1",
        "notes": "v1 scenario assumes full closure of Hormuz.",
    },
    {
        "parameter": "total_oil_flow_million_bpd_2024",
        "value": "20.261741721311477",
        "units": "million_bpd",
        "source_id": "eia_hormuz_fig1_2025_06_16_xlsx",
        "notes": "Total oil flow through Hormuz in 2024.",
    },
    {
        "parameter": "crude_and_condensate_flow_million_bpd_2024",
        "value": "14.318613808743168",
        "units": "million_bpd",
        "source_id": "eia_hormuz_fig1_2025_06_16_xlsx",
        "notes": "Crude and condensate flow through Hormuz in 2024.",
    },
    {
        "parameter": "petroleum_products_flow_million_bpd_2024",
        "value": "5.9431279125683094",
        "units": "million_bpd",
        "source_id": "eia_hormuz_fig1_2025_06_16_xlsx",
        "notes": "Petroleum-products flow through Hormuz in 2024.",
    },
    {
        "parameter": "available_bypass_capacity_million_bpd",
        "value": "2.6",
        "units": "million_bpd",
        "source_id": "eia_hormuz_tie_2025_06_16_html",
        "notes": "EIA estimate of effective unused Saudi and UAE bypass capacity.",
    },
    {
        "parameter": "asian_destination_share_of_crude_flows",
        "value": "0.84",
        "units": "share",
        "source_id": "eia_hormuz_tie_2025_06_16_html",
        "notes": "Share of Hormuz crude and condensate going to Asian markets in 2024.",
    },
    {
        "parameter": "top_four_asian_destination_share_of_crude_flows",
        "value": "0.69",
        "units": "share",
        "source_id": "eia_hormuz_tie_2025_06_16_html",
        "notes": "Share of Hormuz crude and condensate going to China, India, Japan, and South Korea in 2024.",
    },
]


MARKET_ANCHORS_V1 = [
    {
        "anchor_id": "analysis_date",
        "value": "2026-04-06",
        "units": "date",
        "source_id": "modeling_assumption_v1",
        "notes": "Simulation analysis date.",
    },
    {
        "anchor_id": "latest_observed_close_date",
        "value": "2026-04-02",
        "units": "date",
        "source_id": "eia_daily_prices_2026_04_06_html",
        "notes": "Latest EIA daily price close visible on the analysis date page.",
    },
    {
        "anchor_id": "latest_observed_brent_usd_per_bbl",
        "value": "127.61",
        "units": "usd_per_bbl",
        "source_id": "eia_daily_prices_2026_04_06_html",
        "notes": "Brent close published on the EIA daily prices page for 2026-04-02.",
    },
    {
        "anchor_id": "latest_observed_wti_usd_per_bbl",
        "value": "113.23",
        "units": "usd_per_bbl",
        "source_id": "eia_daily_prices_2026_04_06_html",
        "notes": "WTI close published on the EIA daily prices page for 2026-04-02.",
    },
    {
        "anchor_id": "eia_next_two_month_floor_brent_usd_per_bbl",
        "value": "95.0",
        "units": "usd_per_bbl",
        "source_id": "eia_steo_2026_03_pdf",
        "notes": "Manual transcription of the STEO statement that Brent stays above $95/b over the next two months.",
    },
    {
        "anchor_id": "eia_q2_2026_average_brent_usd_per_bbl",
        "value": "91.0",
        "units": "usd_per_bbl",
        "source_id": "eia_steo_2026_03_pdf",
        "notes": "Manual transcription of the STEO 2Q26 Brent average under Hormuz disruption.",
    },
    {
        "anchor_id": "opec_may_2026_supply_adjustment_million_bpd",
        "value": "0.206",
        "units": "million_bpd",
        "source_id": "opec_2026_04_05_production_decision_html",
        "notes": "May 2026 production adjustment announced on 2026-04-05.",
    },
]


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)


def download(url: str) -> bytes:
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is required to fetch Hormuz source snapshots")
    result = subprocess.run(
        [
            curl,
            "-L",
            "--max-time",
            "60",
            "-A",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
            url,
        ],
        check=True,
        capture_output=True,
    )
    return result.stdout


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def xlsx_rows(payload: bytes) -> list[list[str]]:
    shared_strings: list[str] = []
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            namespace = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            for item in root.findall("a:si", namespace):
                text = "".join(
                    node.text or "" for node in item.findall(".//a:t", namespace)
                )
                shared_strings.append(text)

        worksheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        namespace = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        rows: list[list[str]] = []
        for row in worksheet.findall(".//a:sheetData/a:row", namespace):
            values: list[str] = []
            for cell in row.findall("a:c", namespace):
                cell_type = cell.attrib.get("t")
                value = cell.find("a:v", namespace)
                if value is None or value.text is None:
                    values.append("")
                elif cell_type == "s":
                    values.append(shared_strings[int(value.text)])
                else:
                    values.append(value.text)
            rows.append(values)
        return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_long_series(rows: list[list[str]], start: int, stop: int, entity_column: str) -> list[dict[str, str]]:
    header = rows[start + 1]
    periods = header[1:7]
    output: list[dict[str, str]] = []
    for raw_row in rows[start + 2 : stop]:
        if not raw_row or not raw_row[0] or raw_row[0].startswith("Data source"):
            continue
        entity = raw_row[0]
        for period, value in zip(periods, raw_row[1:7]):
            output.append(
                {
                    entity_column: entity,
                    "period": period,
                    "million_bpd": value,
                }
            )
    return output


def build_processed_tables(fig1_rows: list[list[str]], fig3_rows: list[list[str]]) -> None:
    total_rows = build_long_series(fig1_rows, 0, 5, "series")
    origin_rows = build_long_series(fig3_rows, 0, 9, "entity")
    destination_rows = build_long_series(fig3_rows, 11, 22, "entity")

    write_csv(
        PROCESSED_DIR / "hormuz_total_flows.csv",
        ["series", "period", "million_bpd"],
        total_rows,
    )
    write_csv(
        PROCESSED_DIR / "hormuz_origin_flows.csv",
        ["entity", "period", "million_bpd"],
        origin_rows,
    )
    write_csv(
        PROCESSED_DIR / "hormuz_destination_flows.csv",
        ["entity", "period", "million_bpd"],
        destination_rows,
    )
    write_csv(
        PROCESSED_DIR / "hormuz_node_set_v1.csv",
        [
            "node_id",
            "label",
            "role",
            "baseline_flow_million_bpd_2024",
            "selection_basis",
            "notes",
        ],
        NODE_SET_V1,
    )
    write_csv(
        PROCESSED_DIR / "hormuz_scenario_parameters_v1.csv",
        ["parameter", "value", "units", "source_id", "notes"],
        SCENARIO_PARAMETERS_V1,
    )
    write_csv(
        PROCESSED_DIR / "market_anchors_v1.csv",
        ["anchor_id", "value", "units", "source_id", "notes"],
        MARKET_ANCHORS_V1,
    )


def main() -> None:
    ensure_dirs()

    manifest_rows: list[dict[str, str]] = []
    checksum_lines: list[str] = []
    raw_payloads: dict[str, bytes] = {}

    for spec in SOURCES:
        payload = download(spec.url)
        raw_path = RAW_DIR / spec.raw_filename
        raw_path.write_bytes(payload)
        digest = sha256_bytes(payload)
        raw_payloads[spec.source_id] = payload
        manifest_rows.append(
            {
                "source_id": spec.source_id,
                "title": spec.title,
                "organization": spec.organization,
                "publication_date": spec.publication_date,
                "access_date": ACCESS_DATE,
                "url": spec.url,
                "local_raw_path": str(raw_path.relative_to(PROBLEM_ROOT)),
                "sha256": digest,
                "notes": spec.notes,
            }
        )
        checksum_lines.append(f"{digest}  {raw_path.relative_to(PROBLEM_ROOT)}")

    write_csv(
        SOURCES_DIR / "source_manifest.csv",
        [
            "source_id",
            "title",
            "organization",
            "publication_date",
            "access_date",
            "url",
            "local_raw_path",
            "sha256",
            "notes",
        ],
        manifest_rows,
    )
    (SOURCES_DIR / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )

    fig1_rows = xlsx_rows(raw_payloads["eia_hormuz_fig1_2025_06_16_xlsx"])
    fig3_rows = xlsx_rows(raw_payloads["eia_hormuz_fig3_2025_06_16_xlsx"])
    build_processed_tables(fig1_rows, fig3_rows)

    print("Fetched raw sources and built processed tables under:")
    print(PROBLEM_ROOT)


if __name__ == "__main__":
    main()
