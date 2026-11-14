# Working in organon-vision (for AI agents)

**How we write code:** [organon/docs/ENGINEERING.md](https://github.com/Philipjj609/organon/blob/main/docs/ENGINEERING.md). Read it before changing anything. Design and decisions: [docs/DESIGN.md](docs/DESIGN.md).

**Layout:**
- `crates/vision-core` (`organon-vision-core`): pure logic, with no file access, network or printing.
  - `sense`: bytes → checked image (the `Fetch` trait), alpha over white.
  - `measure`: look, ink, dhash. Shared with organon-video; `golden_*` pins the numbers.
  - `palette`, `text` (`TextReader`, line splitting), `objects` (`ObjectDetector`, YOLO decode), `layout`.
  - `engines` (feature `models`): ocrs and YOLO on RTen, from model bytes.
  - `perceive`: the flow; `emit`: protocol messages.
- `crates/vision-cli` (`organon-vision`): the binary. `net` is the only network code.

**Quality gate** (CI on Linux, macOS, Windows): `cargo fmt --all --check && cargo clippy --all-targets --all-features -- -D warnings && cargo test`. Every bug fix gets a regression test.

**Rules:**
1. Perception does not interpret: roles are `text` or `figure`; labels are the model's, with scores.
2. Regions are pixels of the original image, `[x, y, w, h]`.
3. Changing `measure` changes organon-video's ids: update the golden test only on purpose.
4. Never commit third-party images or model weights. Fixtures are drawn in code.

**Try it:** `organon-vision models`, then `organon-vision see <a screenshot>`.
