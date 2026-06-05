use crate::problems::multi_echelon::verification::{
    gijs_relative_reference_instances, gijs_relative_verification_summary,
    published_constant_base_stock_reference_instances, van_roy_reproduction_summary,
    DEFAULT_GIJS_RELATIVE_VERIFICATION_SEED, GIJS_RELATIVE_VERIFICATION_METRIC,
    VAN_ROY_REPRODUCTION_METRIC,
};

#[test]
fn gijs_relative_reference_rows_are_present_and_frozen() {
    let references = gijs_relative_reference_instances();
    assert_eq!(references.len(), 2);
    assert_eq!(references[0].name, "van_roy1997_case_study1");
    assert_eq!(references[1].name, "van_roy1997_case_study2");
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
    assert!(summary.literature_reference_present);
    assert!(!summary.implementation_literature_verified);
    assert_eq!(
        summary.literature_verification_metric,
        GIJS_RELATIVE_VERIFICATION_METRIC
    );
    assert_eq!(summary.literature_verification_target_count, 2);
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
    assert!(summary.all_published_constant_base_stock_rows_reproduced_within_tolerance);
    assert!(!summary.repo_generates_published_relative_rows);
    assert!(!summary.can_mark_literature_verified);
}

#[test]
fn van_roy_reproduction_summary_checks_absolute_published_rows() {
    let references = published_constant_base_stock_reference_instances();
    assert_eq!(references.len(), 3);
    assert_eq!(
        references[0].published_constant_base_stock_levels,
        &[10, 16]
    );
    assert_eq!(
        references[1].published_constant_base_stock_levels,
        &[330, 23]
    );
    assert_eq!(
        references[2].published_constant_base_stock_levels,
        &[460, 22]
    );

    let summary = van_roy_reproduction_summary(2, DEFAULT_GIJS_RELATIVE_VERIFICATION_SEED)
        .expect("summary must build");

    assert_eq!(summary.rows.len(), 3);
    assert!(summary.literature_reference_present);
    assert_eq!(
        summary.literature_verification_metric,
        VAN_ROY_REPRODUCTION_METRIC
    );
    assert!(!summary.implementation_literature_verified);
    assert!(summary.all_published_constant_base_stock_rows_reproduced_within_tolerance);
    assert_eq!(summary.rows[1].instance_name, "van_roy1997_case_study1");
    assert!(
        summary.rows[1]
            .repo_gap_vs_published_constant_cost_pct
            .abs()
            > 1.0
    );
}
