# hormuz_strait/flownet

This folder records the Strait of Hormuz disruption model in the shared FlowNet language.

It is the source-backed physical and control formulation that the executable month-ahead price
scenario engine now uses.

## The Seven Questions

1. What inventory states exist?

- exporter-side oil supply available to enter the maritime network
- destination-market inventories and working stocks
- a Gulf refining and storage hub that can absorb some local crude flow
- a strategic reserve and floating storage buffer that can release oil during disruption
- unserved market demand carried as shortage or rationing state

2. How can material move or transform?

- Gulf exporters ship oil into the Strait of Hormuz transit lane
- some flow can be rerouted into aggregate Saudi and UAE bypass capacity
- rerouted and normal flows move through open-water delivery lanes to destination markets
- destination inventories serve market demand
- strategic reserves can be released into destination markets when disruption is active

3. What random events occur?

- destination-market demand varies over time
- the chokepoint can experience disruption onset, persistence, and reopening
- rerouting around the Strait can introduce additional transit delay and congestion

4. What can the controller choose?

- how much exporter flow to reroute into available bypass capacity
- how much reserve inventory to release
- how to allocate scarce delivered oil across destination markets

5. What can the controller observe, and when?

- the current disruption state of the Strait
- exporter supply and delivered inventories
- reserve volumes
- market-demand weights from the 2024 destination pattern
- available bypass capacity

6. How is performance scored?

- shortage or rationing cost at destination markets
- reserve release and emergency fulfillment cost
- rerouting cost
- inventory carrying cost
- custom price-shock penalty for severe unmet demand

7. What timing rules and constraints shape the system?

- the controller observes closure status and available inventories
- rerouting and reserve-release decisions are made
- the disruption state limits Hormuz transit
- deliveries arrive into destination-market inventories
- demand is realized and served
- shortages, reserve costs, and congestion costs are charged

This formulation uses a `20`-node scenario because a Hormuz model has to keep the physical
chokepoint and the main destination blocs visible in the same network.
