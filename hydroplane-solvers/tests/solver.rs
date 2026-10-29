//! [`Solver`] end to end on the CPU: it meets the tolerance, and when nothing can, it tries what
//! there is and says so.

use hydroplane_solvers::Solver;
use hydroplane_solvers::auto::{Device, Method, Options, Reason, Verify};

const SIDE: usize = 12;

/// Lower triangle of the 7-point Laplacian on a `SIDE³` grid with a small diagonal shift, and a
/// right-hand side.
fn laplacian() -> (Vec<usize>, Vec<usize>, Vec<f64>, Vec<f64>) {
    let at = |x: usize, y: usize, z: usize| (z * SIDE + y) * SIDE + x;
    let (mut col_ptr, mut row_idx, mut val) = (vec![0], Vec::new(), Vec::new());
    for z in 0..SIDE {
        for y in 0..SIDE {
            for x in 0..SIDE {
                row_idx.push(at(x, y, z));
                val.push(6.0 + 1e-3);
                for (next, inside) in [(at(x + 1, y, z), x + 1 < SIDE), (at(x, y + 1, z), y + 1 < SIDE), (at(x, y, z + 1), z + 1 < SIDE)] {
                    if inside {
                        row_idx.push(next);
                        val.push(-1.0);
                    }
                }
                col_ptr.push(row_idx.len());
            }
        }
    }
    let b = (0..SIDE * SIDE * SIDE).map(|i| ((i * 7919) % 101) as f64 - 50.0).collect();
    (col_ptr, row_idx, val, b)
}

#[test]
fn meets_the_tolerance_on_the_cpu() {
    let (col_ptr, row_idx, val, b) = laplacian();
    let options = Options { device: Some(Device::Cpu), tolerance: 1e-8, ..Default::default() };
    let mut solver = Solver::with_options(&col_ptr, &row_idx, options).unwrap();
    assert_eq!(solver.plan().reason, Reason::Tiny, "{}", solver.plan());
    solver.factorize(&val).unwrap();
    let mut x = b.clone();
    let report = solver.solve_in_place(&mut x).unwrap();
    assert!(report.verified && report.meets_tolerance == Some(true), "{report:?}");
}

#[test]
fn when_nothing_can_meet_the_tolerance_it_says_so() {
    let (col_ptr, row_idx, val, b) = laplacian();
    let options = Options { device: Some(Device::Cpu), tolerance: 1e-20, verify: Verify::Always, ..Default::default() };
    let mut solver = Solver::with_options(&col_ptr, &row_idx, options).unwrap();
    solver.factorize(&val).unwrap();
    let mut x = b.clone();
    let report = solver.solve_in_place(&mut x).unwrap();
    assert_eq!(report.meets_tolerance, Some(false));
    assert!(report.residual.unwrap() < 1e-12, "the best effort is still a solution: {report:?}");
}

#[test]
fn a_pinned_strategy_is_kept_even_when_it_misses() {
    let (col_ptr, row_idx, val, b) = laplacian();
    let options = Options { device: Some(Device::Cpu), method: Some(Method::Pcg), precision: Some(hydroplane_solvers::Precision::F64), tolerance: 1e-20, ..Default::default() };
    let mut solver = Solver::with_options(&col_ptr, &row_idx, options).unwrap();
    solver.factorize(&val).unwrap();
    let mut x = b.clone();
    let report = solver.solve_in_place(&mut x).unwrap();
    assert_eq!((report.strategy.method, report.meets_tolerance), (Method::Pcg, Some(false)));
}

#[test]
fn a_solve_before_a_factorization_is_an_error() {
    let (col_ptr, row_idx, _, b) = laplacian();
    let mut solver = Solver::new(&col_ptr, &row_idx).unwrap();
    let mut x = b;
    assert!(solver.solve_in_place(&mut x).is_err());
}

#[test]
fn updated_values_are_solved_exactly_with_the_old_factor() {
    let (col_ptr, row_idx, val, b) = laplacian();
    let changed: Vec<f64> = val.iter().enumerate().map(|(p, v)| v * (1.0 + 0.05 * ((p * 7919) % 13) as f64 / 13.0)).collect();
    let options = Options { device: Some(Device::Cpu), tolerance: 1e-8, verify: Verify::Always, ..Default::default() };
    let mut solver = Solver::with_options(&col_ptr, &row_idx, options).unwrap();
    solver.factorize(&val).unwrap();
    solver.update_values(&changed).unwrap();
    let mut x = b.clone();
    let report = solver.solve_in_place(&mut x).unwrap();
    assert_eq!(report.strategy.method, Method::Direct, "the factor is kept");
    assert!(report.meets_tolerance == Some(true) && report.iterations > 0 && report.iterations < 20, "{report:?}");
}

#[cfg(target_os = "macos")]
#[test]
fn updated_values_with_an_accelerate_factor() {
    use hydroplane_solvers::auto::Library;
    let (col_ptr, row_idx, val, b) = laplacian();
    let changed: Vec<f64> = val.iter().enumerate().map(|(p, v)| v * (1.0 + 0.05 * ((p * 7919) % 13) as f64 / 13.0)).collect();
    let options = Options { device: Some(Device::Cpu), library: Some(Library::Accelerate), tolerance: 1e-8, verify: Verify::Always, ..Default::default() };
    let mut solver = Solver::with_options(&col_ptr, &row_idx, options).unwrap();
    assert_eq!(solver.plan().strategy.library, Library::Accelerate, "{}", solver.plan());
    solver.factorize(&val).unwrap();
    solver.update_values(&changed).unwrap();
    let mut x = b.clone();
    let report = solver.solve_in_place(&mut x).unwrap();
    assert!(report.meets_tolerance == Some(true) && report.iterations > 0 && report.iterations < 20, "{report:?}");
}

/// Lower triangle of `Σₑ sₑ Kₑ + 10⁻³ I` on a `side³` grid of hexahedra, `Kₑ` the graph Laplacian
/// of an element's eight nodes: what changing each element's stiffness by `sₑ` does to an FEM
/// tangent, spectrally within `[min sₑ, max sₑ]` of the unscaled one.
fn elements(side: usize, scale: impl Fn(usize) -> f64) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let nodes = side + 1;
    let at = |x: usize, y: usize, z: usize| (z * nodes + y) * nodes + x;
    let n = nodes * nodes * nodes;
    let mut entries = std::collections::BTreeMap::<(usize, usize), f64>::new();
    for i in 0..n {
        entries.insert((i, i), 1e-3);
    }
    for e in 0..side * side * side {
        let (x, y, z) = (e % side, e / side % side, e / (side * side));
        let corners: Vec<usize> = (0..8).map(|c| at(x + (c & 1), y + (c >> 1 & 1), z + (c >> 2))).collect();
        let s = scale(e);
        for &i in &corners {
            for &j in &corners {
                if i >= j {
                    *entries.entry((i, j)).or_default() += s * if i == j { 7.0 } else { -1.0 };
                }
            }
        }
    }
    let mut by_column: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for (&(i, j), &v) in &entries {
        by_column[j].push((i, v));
    }
    let (mut col_ptr, mut row_idx, mut val) = (vec![0], Vec::new(), Vec::new());
    for column in by_column {
        for (i, v) in column {
            row_idx.push(i);
            val.push(v);
        }
        col_ptr.push(row_idx.len());
    }
    (col_ptr, row_idx, val)
}

#[test]
fn a_drifting_tangent_is_solved_in_a_few_iterations_on_the_old_factor() {
    let base = |e: usize| 1.0 + ((e * 7919) % 17) as f64 / 17.0;
    let (col_ptr, row_idx, val) = elements(10, base);
    let b: Vec<f64> = (0..col_ptr.len() - 1).map(|i| ((i * 7919) % 101) as f64 - 50.0).collect();
    for drift in [0.05, 0.2] {
        let (_, _, changed) = elements(10, |e| base(e) * (1.0 + drift * ((e as f64) * 0.37).sin()));
        let options = Options { device: Some(Device::Cpu), method: Some(Method::Direct), tolerance: 1e-8, verify: Verify::Always, ..Default::default() };
        let mut solver = Solver::with_options(&col_ptr, &row_idx, options).unwrap();
        solver.factorize(&val).unwrap();
        solver.update_values(&changed).unwrap();
        let mut x = b.clone();
        let report = solver.solve_in_place(&mut x).unwrap();
        let bound = if drift < 0.1 { 10 } else { 20 };
        eprintln!("drift {drift}: {} iterations, residual {:?}", report.iterations, report.residual);
        assert!(report.meets_tolerance == Some(true) && report.iterations <= bound, "drift {drift}: {report:?}");
    }
}
