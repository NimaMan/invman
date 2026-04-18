#[derive(Clone, Debug, PartialEq)]
pub struct ClassicBeerGameBenchmarkSummary {
    pub per_agent_costs: [f64; 4],
    pub total_cost: f64,
}

fn round_half_away_from_zero(value: f64) -> usize {
    if value <= 0.0 {
        0
    } else {
        (value + 0.5).floor() as usize
    }
}

pub fn simulate_classic_sterman_benchmark() -> ClassicBeerGameBenchmarkSummary {
    let mut end_customer_demands = [0usize; 38];
    end_customer_demands[2..6].fill(4);
    end_customer_demands[6..38].fill(8);

    let mut eecd = 4.0;
    let mut eo_r = 4.0;
    let mut eo_w = 4.0;
    let mut eo_d = 4.0;

    let mut b = [0usize; 4];
    let mut i = [12usize; 4];

    let mut iti1 = [4usize; 3];
    let mut iti2 = [4usize; 3];
    let mut wipi1 = 4usize;
    let mut wipi2 = 4usize;

    let mut o_r = [0usize; 39];
    let mut o_w = [0usize; 39];
    let mut o_d = [0usize; 39];
    let mut psr = [0usize; 39];
    o_r[2] = 4;
    o_w[2] = 4;
    o_d[2] = 4;
    psr[2] = 4;

    let mut io_w = [0usize; 39];
    let mut io_d = [0usize; 39];
    let mut io_f = [0usize; 39];
    io_w[2] = 4;
    io_d[2] = 4;
    io_f[2] = 4;

    let mut tc = [0.0; 4];
    let s_prime = [28.0, 28.0, 28.0, 20.0];

    for t in 2..=37 {
        i[0] += iti2[0];
        i[1] += iti2[1];
        i[2] += iti2[2];
        i[3] += wipi2;

        iti2 = iti1;
        wipi2 = wipi1;
        iti1 = [0; 3];

        let s_endc = i[0].min(b[0] + end_customer_demands[t]);
        let s_r = i[1].min(b[1] + io_w[t]);
        let s_w = i[2].min(b[2] + io_d[t]);
        let s_d = i[3].min(b[3] + io_f[t]);

        iti1[0] = s_r;
        iti1[1] = s_w;
        iti1[2] = s_d;
        wipi1 = psr[t];

        let total_demand_r = b[0] + end_customer_demands[t];
        let total_demand_w = b[1] + io_w[t];
        let total_demand_d = b[2] + io_d[t];
        let total_demand_f = b[3] + io_f[t];

        b[0] = total_demand_r - s_endc;
        b[1] = total_demand_w - s_r;
        b[2] = total_demand_d - s_w;
        b[3] = total_demand_f - s_d;

        i[0] -= s_endc;
        i[1] -= s_r;
        i[2] -= s_w;
        i[3] -= s_d;

        // Optimal Sterman benchmark uses theta = 0, so the expectations stay constant at 4.
        let _ = (&mut eecd, &mut eo_r, &mut eo_w, &mut eo_d);

        io_w[t + 1] = o_r[t];
        io_d[t + 1] = o_w[t];
        io_f[t + 1] = o_d[t];

        let ei = [
            i[0] as i32 - b[0] as i32,
            i[1] as i32 - b[1] as i32,
            i[2] as i32 - b[2] as i32,
            i[3] as i32 - b[3] as i32,
        ];

        let sl_r = io_w[t + 1] + b[1] + iti1[0] + iti2[0];
        let sl_w = io_d[t + 1] + b[2] + iti1[1] + iti2[1];
        let sl_d = io_f[t + 1] + b[3] + iti1[2] + iti2[2];
        let sl_f = wipi1 + wipi2;

        if t <= 5 {
            o_r[t + 1] = 4;
            o_w[t + 1] = 4;
            o_d[t + 1] = 4;
            psr[t + 1] = 4;
        } else {
            o_r[t + 1] =
                round_half_away_from_zero((eecd + (s_prime[0] - ei[0] as f64 - sl_r as f64)).max(0.0));
            o_w[t + 1] =
                round_half_away_from_zero((eo_r + (s_prime[1] - ei[1] as f64 - sl_w as f64)).max(0.0));
            o_d[t + 1] =
                round_half_away_from_zero((eo_w + (s_prime[2] - ei[2] as f64 - sl_d as f64)).max(0.0));
            psr[t + 1] =
                round_half_away_from_zero((eo_d + (s_prime[3] - ei[3] as f64 - sl_f as f64)).max(0.0));
        }

        for agent_idx in 0..4 {
            tc[agent_idx] += b[agent_idx] as f64 + 0.5 * i[agent_idx] as f64;
        }
    }

    ClassicBeerGameBenchmarkSummary {
        per_agent_costs: [tc[0], tc[1], tc[2], tc[3]],
        total_cost: tc.iter().sum(),
    }
}
