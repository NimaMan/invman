# maritime_traffic

Reproducible home for ship-traffic and passage data that can refine the Hormuz FlowNet.

This folder is specifically for:

- vessel transit counts through the Strait of Hormuz before and after the current regional war began
- AIS-derived cargo and tanker passage snapshots
- supporting maritime-security notes that help explain why observed traffic changed

For this module, the primary conflict breakpoint is:

- `2026-02-28`
  - the start of hostilities used by JMIC/MSCIO in the March 2026 advisories

Current structure:

- `data/raw/`
  - downloaded source PDFs
- `data/processed/`
  - compact tables of passage observations we can use in modeling
- `sources/`
  - manifest and checksums for the raw files
- `notes/`
  - interpretation notes and FlowNet-integration guidance
- `scripts/`
  - reserved for later extraction/build automation

Current purpose in FlowNet:

- calibrate time-varying capacity multipliers on the `hormuz_transit_lane`
- distinguish tanker traffic collapse from general cargo collapse
- add an `ais_visibility` or `observed_vs_true_traffic` factor because JMIC repeatedly warns that dark
  transits remain possible
- support war-risk and congestion states around anchorages and approaches

Important boundary:

- Bab el-Mandeb traffic in these JMIC notes is useful as a regional shipping-risk signal
- it is not a physical substitute route for Gulf-origin crude moving to Asia through Hormuz

