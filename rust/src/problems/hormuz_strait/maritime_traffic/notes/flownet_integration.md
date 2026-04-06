# FlowNet Integration Notes

## Breakpoint

Use `2026-02-28` as the start-of-hostilities breakpoint for this submodule. JMIC/MSCIO explicitly
frames the March 2026 disruption metrics relative to that date.

## What the data says

- Immediate pre-crisis AIS-derived Strait of Hormuz traffic was still active on `2026-02-28`.
- By `2026-03-04`, observed AIS-derived commercial traffic through the Strait had nearly collapsed.
- By `2026-03-15` and `2026-03-17`, traffic remained far below historical averages, with only a few
  observed commercial transits per day.
- JMIC repeatedly notes that dark transits are possible, so observed AIS counts are lower bounds rather
  than complete truth.

## How to use it in FlowNet

1. `hormuz_transit_lane` capacity:
   - derive a daily `observed_capacity_multiplier` from the passage snapshot table
   - example: compare observed crisis traffic to the `~138/day` historical reference

2. Vessel class split:
   - keep separate factors for `cargo` and `tanker`
   - tanker traffic can collapse differently from general cargo traffic

3. Observation uncertainty:
   - add a state variable or observation note for `ais_visibility_gap`
   - JMIC notes that non-AIS or GNSS-degraded traffic can still be present

4. Congestion and risk:
   - use attack intensity, anchorage congestion, and GNSS interference as drivers of effective flow
     reduction, not just hard closure

## What not to do

- Do not treat Bab el-Mandeb transit counts as a physical bypass path for Hormuz oil.
- Do not equate AIS-derived counts with total true ship counts without an uncertainty adjustment.

