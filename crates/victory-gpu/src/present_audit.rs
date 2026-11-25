//! Measure the cadence of host DXGI `Present` calls returning `S_OK`.

use std::fmt::Write as _;
use std::time::{Duration, Instant};

use windows::Win32::Foundation::S_OK;
use windows::core::HRESULT;

#[derive(Default)]
pub(crate) struct PresentAudit {
    last_report: Option<Instant>,
    last_accepted: Option<Instant>,
    accepted_total: u64,
    positive_total: u64,
    error_total: u64,
    last_positive: u32,
    last_error: u32,
    intervals: Vec<(u64, Option<Duration>)>,
}

impl PresentAudit {
    pub(crate) fn configured() -> Option<Self> {
        (std::env::var("VICTORY_PRESENT_AUDIT").as_deref() == Ok("1")).then(Self::default)
    }

    pub(crate) fn record(&mut self, frame: u64, result: HRESULT) {
        let now = Instant::now();
        self.record_result(frame, result, now);
        if let Some(intervals_us) = self.take_report(now) {
            tracing::info!(
                target: "victory_present_audit",
                through_frame = frame,
                accepted_total = self.accepted_total,
                positive_total = self.positive_total,
                error_total = self.error_total,
                last_positive = self.last_positive,
                last_error = self.last_error,
                intervals_us = %intervals_us,
                "DXGI present audit"
            );
        }
    }

    fn record_result(&mut self, frame: u64, result: HRESULT, now: Instant) {
        if result == S_OK {
            self.accepted_total += 1;
            let interval = self.last_accepted.replace(now).map(|last| now - last);
            self.intervals.push((frame, interval));
        } else if result.is_ok() {
            self.positive_total += 1;
            self.last_positive = result.0 as u32;
        } else {
            self.error_total += 1;
            self.last_error = result.0 as u32;
        }
    }

    /// Report on the first call, then on the next call at least one second
    /// after each report. Preserve the last accepted timestamp across reports.
    fn take_report(&mut self, now: Instant) -> Option<String> {
        if self
            .last_report
            .is_some_and(|last| now - last < Duration::from_secs(1))
        {
            return None;
        }
        self.last_report = Some(now);
        let mut samples = String::new();
        for (frame, interval) in self.intervals.drain(..) {
            if !samples.is_empty() {
                samples.push(',');
            }
            match interval {
                Some(interval) => write!(samples, "{frame}:{}", interval.as_micros()),
                None => write!(samples, "{frame}:-"),
            }
            .expect("writing to a String cannot fail");
        }
        Some(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Foundation::DXGI_STATUS_OCCLUDED;
    use windows::Win32::Graphics::Dxgi::DXGI_ERROR_DEVICE_REMOVED;

    #[test]
    fn only_s_ok_contributes_an_interval() {
        let mut audit = PresentAudit::default();
        let start = Instant::now();

        audit.record_result(1, DXGI_STATUS_OCCLUDED, start);
        audit.record_result(2, S_OK, start + Duration::from_millis(10));
        audit.record_result(3, DXGI_STATUS_OCCLUDED, start + Duration::from_millis(20));
        audit.record_result(4, DXGI_ERROR_DEVICE_REMOVED, start + Duration::from_millis(40));
        audit.record_result(5, S_OK, start + Duration::from_millis(70));

        assert_eq!(audit.accepted_total, 2);
        assert_eq!(audit.positive_total, 2);
        assert_eq!(audit.error_total, 1);
        assert_eq!(audit.last_positive, DXGI_STATUS_OCCLUDED.0 as u32);
        assert_eq!(audit.last_error, DXGI_ERROR_DEVICE_REMOVED.0 as u32);
        assert_eq!(
            audit.take_report(start + Duration::from_millis(70)),
            Some("2:-,5:60000".to_owned())
        );
    }

    #[test]
    fn reports_preserve_intervals_across_windows() {
        let mut audit = PresentAudit::default();
        let start = Instant::now();

        audit.record_result(1, S_OK, start);
        assert_eq!(audit.take_report(start), Some("1:-".to_owned()));
        let early = start + Duration::from_millis(999);
        audit.record_result(2, S_OK, early);
        assert_eq!(audit.take_report(early), None);
        let due = start + Duration::from_secs(1);
        audit.record_result(3, DXGI_STATUS_OCCLUDED, due);
        assert_eq!(audit.take_report(due), Some("2:999000".to_owned()));
        let next = due + Duration::from_secs(1);
        audit.record_result(4, S_OK, next);
        assert_eq!(audit.take_report(next), Some("4:1001000".to_owned()));
        assert_eq!(audit.accepted_total, 3);
    }
}
