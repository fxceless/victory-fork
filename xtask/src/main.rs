//! Repository maintenance. Run with `cargo xtask <command>`.

mod policy;

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "xtask")]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Enforce the asset policy: the repository holds only small text files
    /// we wrote. Checks tracked and untracked-but-not-ignored files.
    Policy {
        /// Check only what is staged for commit (used by the pre-commit hook).
        #[arg(long)]
        staged: bool,
    },
    /// Point git at .githooks/ so every commit runs the policy check.
    InstallHooks,
    /// Run what CI runs, in CI's order: the policy check, clippy for the
    /// plain and the diagnostics build, then the tests. Stops at the first
    /// failure.
    Check {
        /// Leave out the tests (clippy and the policy check only).
        #[arg(long)]
        no_test: bool,
    },
}

fn main() -> Result<()> {
    let root = workspace_root();
    match Cli::parse().command {
        Cmd::Policy { staged } => policy::run(&root, staged),
        Cmd::InstallHooks => {
            let status = Command::new("git")
                .args(["config", "core.hooksPath", ".githooks"])
                .current_dir(&root)
                .status()
                .context("running git")?;
            anyhow::ensure!(
                status.success(),
                "git config failed (is this a git repository?)"
            );
            println!("hooks installed: commits now run `cargo xtask policy --staged`");
            Ok(())
        }
        Cmd::Check { no_test } => check(&root, no_test),
    }
}

/// Keep the steps in step with `.github/workflows/ci.yml`.
fn check(root: &Path, no_test: bool) -> Result<()> {
    policy::run(root, false)?;
    let clippy = ["clippy", "--workspace", "--all-targets", "--locked"];
    let deny = ["--", "-D", "warnings"];
    let mut steps: Vec<(&str, Vec<&str>)> = vec![
        ("clippy", [&clippy[..], &deny].concat()),
        (
            "clippy, diagnostics build",
            [&clippy[..], &["--all-features"], &deny].concat(),
        ),
    ];
    if !no_test {
        steps.push(("tests", vec!["test", "--workspace", "--locked"]));
    }
    // `cargo xtask` sets CARGO to the cargo that ran it (same toolchain).
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    for (name, args) in steps {
        println!("== {name}: cargo {}", args.join(" "));
        let status = Command::new(&cargo)
            .args(&args)
            .current_dir(root)
            .status()
            .context("running cargo")?;
        anyhow::ensure!(status.success(), "{name} failed");
    }
    println!("all checks passed");
    Ok(())
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives in the workspace")
        .into()
}
