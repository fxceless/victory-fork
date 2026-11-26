//! Reproducible summaries of the diagnostic frame log.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

#[derive(Clone, Copy, Debug, PartialEq)]
struct Sample {
    milliseconds: f64,
    draws: f64,
}

fn parse(line: &str) -> Option<Sample> {
    let rest = line.split_once("usual frame (the mean of ")?.1;
    let (_, rest) = rest.split_once("): ")?;
    let milliseconds = rest.split_once(" ms |")?.0.parse().ok()?;
    let draws = rest
        .split(" | ")
        .find_map(|part| part.strip_suffix(" draws"))?
        .parse()
        .ok()?;
    Some(Sample {
        milliseconds,
        draws,
    })
}

#[derive(Debug, PartialEq)]
struct Summary {
    count: usize,
    median_ms: f64,
    p90_ms: f64,
    intercept_ms: f64,
    ms_per_1000_draws: f64,
    r_squared: f64,
}

fn percentile(sorted: &[f64], numerator: usize, denominator: usize) -> f64 {
    sorted[(sorted.len() - 1) * numerator / denominator]
}

fn summarise(samples: &[Sample]) -> Result<Summary> {
    if samples.len() < 2 {
        bail!("need at least two matching frame samples");
    }
    let mut times: Vec<f64> = samples.iter().map(|s| s.milliseconds).collect();
    times.sort_by(f64::total_cmp);

    let n = samples.len() as f64;
    let mean_x = samples.iter().map(|s| s.draws).sum::<f64>() / n;
    let mean_y = samples.iter().map(|s| s.milliseconds).sum::<f64>() / n;
    let xx = samples
        .iter()
        .map(|s| (s.draws - mean_x).powi(2))
        .sum::<f64>();
    if xx == 0.0 {
        bail!("draw counts do not vary; cannot fit frame time against draws");
    }
    let xy = samples
        .iter()
        .map(|s| (s.draws - mean_x) * (s.milliseconds - mean_y))
        .sum::<f64>();
    let slope = xy / xx;
    let intercept = mean_y - slope * mean_x;
    let total = samples
        .iter()
        .map(|s| (s.milliseconds - mean_y).powi(2))
        .sum::<f64>();
    let residual = samples
        .iter()
        .map(|s| (s.milliseconds - (intercept + slope * s.draws)).powi(2))
        .sum::<f64>();

    Ok(Summary {
        count: samples.len(),
        median_ms: percentile(&times, 1, 2),
        p90_ms: percentile(&times, 9, 10),
        intercept_ms: intercept,
        ms_per_1000_draws: slope * 1000.0,
        r_squared: if total == 0.0 {
            1.0
        } else {
            1.0 - residual / total
        },
    })
}

fn analyse(path: &Path, min_draws: u64) -> Result<Summary> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let samples: Vec<_> = text
        .lines()
        .filter_map(parse)
        .filter(|s| s.draws >= min_draws as f64)
        .collect();
    summarise(&samples).with_context(|| format!("analysing {}", path.display()))
}

pub(crate) fn run(paths: &[PathBuf], min_draws: u64) -> Result<()> {
    for path in paths {
        let s = analyse(path, min_draws)?;
        println!("{}", path.display());
        println!(
            "  samples {} | median {:.1} ms | p90 {:.1} ms | fit {:.2} ms + {:.3} ms/1k draws | R^2 {:.3}",
            s.count, s.median_ms, s.p90_ms, s.intercept_ms, s.ms_per_1000_draws, s.r_squared
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Sample, parse, summarise};

    #[test]
    fn parses_a_real_frame_line_shape() {
        let line = "time INFO victory_gpu::framelog: usual frame (the mean of 36 not long): 93 ms | render thread: ran 5 ms | 2676 draws | 460 MB free";
        assert_eq!(
            parse(line),
            Some(Sample {
                milliseconds: 93.0,
                draws: 2676.0
            })
        );
        assert_eq!(parse("ordinary log line"), None);
    }

    #[test]
    fn fits_time_by_draw_count() {
        let samples = [
            Sample {
                draws: 1000.0,
                milliseconds: 30.0,
            },
            Sample {
                draws: 3000.0,
                milliseconds: 50.0,
            },
            Sample {
                draws: 2000.0,
                milliseconds: 40.0,
            },
        ];
        let s = summarise(&samples).unwrap();
        assert_eq!(s.count, 3);
        assert_eq!(s.median_ms, 40.0);
        assert_eq!(s.intercept_ms, 20.0);
        assert_eq!(s.ms_per_1000_draws, 10.0);
        assert_eq!(s.r_squared, 1.0);
    }
}
