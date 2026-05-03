use crate::problems::multi_echelon::verification::{
    gijs_relative_reference_instances, gijs_relative_verification_summary,
    DEFAULT_GIJS_RELATIVE_VERIFICATION_SEED,
};

#[test]
fn gijs_relative_reference_rows_are_present_and_frozen() {
    let references = gijs_relative_reference_instances();
    assert_eq!(references.len(), 2);
    assert_eq!(references[0].name, "gijsbrechts2022_setting1");
    assert_eq!(references[1].name, "gijsbrechts2022_setting2");
    assert_eq!(
        references[0].published_constant_base_stock_levels,
        &[330, 23]
    );
    assert_eq!(
        references[1].published_constant_base_stock_levels,
        &[460, 22]
    );
    assert_eq!(references[0].published_a3c_savings_pct, Some(8.95));
    assert_eq!(references[1].published_a3c_savings_pct, Some(12.09));
}

#[test]
fn gijs_relative_verification_summary_computes_carried_implied_costs() {
    let summary = gijs_relative_verification_summary(2, DEFAULT_GIJS_RELATIVE_VERIFICATION_SEED)
        .expect("summary must build");

    assert_eq!(summary.rows.len(), 2);
    assert_eq!(summary.source.contains("Gijsbrechts"), true);
    assert!((summary.rows[0].published_a3c_implied_mean_cost - 1185.471).abs() < 1e-9);
    assert!((summary.rows[1].published_a3c_implied_mean_cost - 1273.8159).abs() < 1e-9);
    assert!((summary.rows[0].published_van_roy_implied_mean_cost - 1171.8).abs() < 1e-9);
    assert!((summary.rows[1].published_van_roy_implied_mean_cost - 1304.1).abs() < 1e-9);
}

#[test]
fn gijs_relative_verification_summary_is_honest_about_current_status() {
    let summary = gijs_relative_verification_summary(2, DEFAULT_GIJS_RELATIVE_VERIFICATION_SEED)
        .expect("summary must build");

    assert!(summary
        .rows
        .iter()
        .all(|row| row.repo_published_constant_base_stock_mean_cost.is_finite()));
    assert!(summary.mean_repo_gap_vs_published_constant_cost.is_finite());
    assert!(!summary.all_published_constant_base_stock_rows_reproduced_within_tolerance);
    assert!(!summary.repo_generates_published_relative_rows);
    assert!(!summary.can_mark_literature_verified);
}
