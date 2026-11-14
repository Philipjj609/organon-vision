//! HTTP behind the core's `Fetch` trait: the only network code.

use std::io::Read as _;
use std::time::Duration;

use organon_vision_core::sense::Fetch;

const USER_AGENT: &str =
    "Mozilla/5.0 (compatible; organon-vision/0.1; +https://github.com/Philipjj609/organon-vision)";
/// Images are a few MB; the object models are tens of MB.
const MAX_BYTES: u64 = 200_000_000;

/// Real HTTP: 60 s timeout, 200 MB cap.
pub struct Http {
    agent: ureq::Agent,
}

impl Http {
    pub fn new() -> Self {
        Self {
            agent: ureq::AgentBuilder::new()
                .timeout(Duration::from_secs(60))
                .user_agent(USER_AGENT)
                .build(),
        }
    }
}

impl Fetch for Http {
    fn get(&self, url: &str) -> Result<Vec<u8>, String> {
        let resp = self.agent.get(url).call().map_err(|e| e.to_string())?;
        let mut body = Vec::new();
        resp.into_reader()
            .take(MAX_BYTES)
            .read_to_end(&mut body)
            .map_err(|e| e.to_string())?;
        Ok(body)
    }
}
