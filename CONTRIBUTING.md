# Contributing

Thanks for wanting to help.
This is a work-in-progress fork of [VIctory](https://github.com/Marvisak/VIctory) that tries to get Xbox Series titles running on a Windows 11.

A few ground rules:

## No game files, no real game names

Never add game files, assets, keys or dumps, and don't link to where they can be found.

Please don't use the real names of copyrighted games in issues, PRs, commits, code or filenames. 
Use a nickname everyone will still recognise, like **Open World Game**. If it's not obvious, do not worry. "The title" always works too.
Same goes for screenshots of game content: describe what you see

## Using AI

We use AI tools on this project ourselves, so this isn't about banning them. It's about knowing what we're reviewing.

- **Say which model you used.** Add a trailer to your commits, e.g. `Assisted-by: Claude Code (claude-opus-5-5)`, and mention it in the PR. Use the real model name, not just "an LLM".
- **Say so if it did most of the work.** If a model wrote most of the diff or came up with the core idea, mention it. That's fine, but it gets a closer look, and you need to be able to explain every line and show you ran it yourself. This matters most for shader translation, memory and sync code, where a plausible mistake can hang a GPU driver.
- **No guardrail-free or easily broken models.** Abliterated, "uncensored" or no-refusal models, and services that sell jailbreaking as a feature, are not allowed. Not sure about one? Ask first. Maintainers may name specific models here later.
- **Mind the licences.** Don't submit anything copied from incompatible code.
- **Hiding AI use** can get a PR closed, however good the code is, and whether or not it was only used to draft or peer review.

## Test feedback goes in the mega-issues

"I tried it on my GPU, here's what happened" is genuinely useful, but please don't open a new issue for it. Post it in the pinned thread instead:

- GPU test feedback: NVIDIA <!-- TODO -->
- GPU test feedback: AMD <!-- TODO -->

That covers frame rates, "works / doesn't work" reports, results from a new build or `VICTORY_*` switch, and driver notes. Standalone issues for these will be closed and pointed back to the right thread.
You may report feedback  to specific PRs on whether the new contribution fixes a certain issue.

A comment is most useful with:

```
Build:    <commit or release>
OS:       <Windows version>
CPU/RAM:  <model / amount>
GPU:      <model / driver version>
Profile:  <Series X or S>
Switches: <VICTORY_* settings, or none>
Result:   <how far it got, fps or ms per frame, what looked wrong>
Crashes:  <relevant crashes.log lines, if any>
```

If you re-test the same setup, edit your old comment rather than adding a new one. Put long logs in a collapsed `<details>` block.

Open a regular issue only when you have something specific and actionable: a reproducible crash or device loss, a bisected commit, or a particular shader that translates wrongly (include its hash). One problem per issue, and search first.

## Code

- `cargo test --workspace` needs no game data and should pass.
- Run `cargo fmt --all` and `cargo clippy --workspace` before pushing. A few lints are relaxed in the workspace `Cargo.toml`; please don't relax more without asking.
- Put experimental behaviour behind an environment switch that's off by default.
- If you approximate what the console does, mark it with a `shortcut:` comment.
- If a change is specific to the title or to one GPU, say so and say what you tested on.
- Keep PRs small and focused, and explain what was broken and how you checked the fix.

Be kind, assume good faith, and keep feedback about the work rather than the person. Thanks again.
