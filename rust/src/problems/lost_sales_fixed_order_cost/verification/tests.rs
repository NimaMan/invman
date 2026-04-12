use crate::problems::lost_sales_fixed_order_cost::literature::{
    BIJVANK_2015_REFERENCE, BIJVANK_2015_TABLE1_REFERENCE,
};
use crate::problems::lost_sales_fixed_order_cost::exact_value_iteration::{
    evaluate_policy, solve_optimal_policy, ExactPolicyKind,
};

#[test]
fn published_reference_row_has_expected_shape() {
    assert!(BIJVANK_2015_REFERENCE.reported_numbers_available);
    assert_eq!(
        BIJVANK_2015_TABLE1_REFERENCE.published_optimal_cost,
        Some(11.46)
    );
    assert_eq!(
        BIJVANK_2015_TABLE1_REFERENCE.published_heuristic_rows.len(),
        3
    );
    assert!(BIJVANK_2015_TABLE1_REFERENCE.literature_verified);
}

#[test]
fn exact_solver_dominates_published_parametric_policies_under_same_cap() {
    let cap = 23;
    let optimal =
        solve_optimal_policy(&BIJVANK_2015_TABLE1_REFERENCE, cap).expect("optimal policy solves");
    let s_s = evaluate_policy(
        &BIJVANK_2015_TABLE1_REFERENCE,
        cap,
        ExactPolicyKind::Ss { s: 17, s_up_to: 23 },
    )
    .expect("(s,S) policy evaluates");
    let s_nq = evaluate_policy(
        &BIJVANK_2015_TABLE1_REFERENCE,
        cap,
        ExactPolicyKind::Snq { s: 17, q: 7 },
    )
    .expect("(s,nQ) policy evaluates");
    let modified = evaluate_policy(
        &BIJVANK_2015_TABLE1_REFERENCE,
        cap,
        ExactPolicyKind::ModifiedSsQ {
            s: 17,
            s_up_to: 23,
            q: 7,
        },
    )
    .expect("modified (s,S,q) policy evaluates");

    assert!(optimal.average_cost <= s_s.average_cost + 1e-9);
    assert!(optimal.average_cost <= s_nq.average_cost + 1e-9);
    assert!(optimal.average_cost <= modified.average_cost + 1e-9);
}

#[test]
fn literature_row_is_matched_tightly_once_cap_is_large_enough() {
    let cap = 24;
    let optimal =
        solve_optimal_policy(&BIJVANK_2015_TABLE1_REFERENCE, cap).expect("optimal policy solves");
    let s_s = evaluate_policy(
        &BIJVANK_2015_TABLE1_REFERENCE,
        cap,
        ExactPolicyKind::Ss { s: 17, s_up_to: 23 },
    )
    .expect("(s,S) policy evaluates");
    let s_nq = evaluate_policy(
        &BIJVANK_2015_TABLE1_REFERENCE,
        cap,
        ExactPolicyKind::Snq { s: 17, q: 7 },
    )
    .expect("(s,nQ) policy evaluates");
    let modified = evaluate_policy(
        &BIJVANK_2015_TABLE1_REFERENCE,
        cap,
        ExactPolicyKind::ModifiedSsQ {
            s: 17,
            s_up_to: 23,
            q: 7,
        },
    )
    .expect("modified (s,S,q) policy evaluates");

    assert_eq!(optimal.first_action, 8);
    assert!((optimal.average_cost - 11.46).abs() < 0.01);
    assert!((s_s.average_cost - 11.62).abs() < 0.01);
    assert!((s_nq.average_cost - 11.56).abs() < 0.01);
    assert!((modified.average_cost - 11.50).abs() < 0.01);
}
