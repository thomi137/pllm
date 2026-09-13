use std::collections::HashMap;

/// Byte-level byte pair encoding.
///
/// IDs 0..=255 are the raw bytes. Every merge creates a new ID from 256 up.
/// Byte-level means: any UTF-8 input is encodable, there is no <UNK>. Umlauts
/// start out as two bytes and grow back together through merges.
#[derive(Debug, Clone, Default)]
pub struct Bpe {
    /// Pair -> rank. Lower rank = learned earlier = applied first.
    ranks: HashMap<(u32, u32), u32>,
    /// Merges in learned order. Index == rank.
    merges: Vec<(u32, u32)>,
    /// ID -> byte sequence, for decode().
    vocab: HashMap<u32, Vec<u8>>,
}

impl Bpe {
    pub fn new() -> Self {
        let mut vocab = HashMap::new();
        for b in 0u32..256 {
            vocab.insert(b, vec![b as u8]);
        }
        Self { ranks: HashMap::new(), merges: Vec::new(), vocab }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Trains `num_merges` merge rules on the corpus.
    ///
    /// Naive O(num_merges * corpus_len) — fine for learning purposes with a
    /// few MB of corpus and a few hundred merges, but run it in a release
    /// build.
    pub fn train(&mut self, corpus: &str, num_merges: u32, verbose: bool) {
        let mut ids: Vec<u32> = corpus.bytes().map(u32::from).collect();

        for i in 0..num_merges {
            let stats = count_pairs(&ids);
            let Some((&pair, &count)) = stats.iter().max_by_key(|(_, &c)| c) else {
                break;
            };
            if count < 2 {
                break; // nothing left to gain
            }

            let new_id = 256 + i;
            ids = merge(&ids, pair, new_id);

            let mut bytes = self.vocab[&pair.0].clone();
            bytes.extend_from_slice(&self.vocab[&pair.1]);

            if verbose {
                println!(
                    "merge {:>4}: ({}, {}) -> {} = {:?}  [{}x]",
                    i,
                    pair.0,
                    pair.1,
                    new_id,
                    String::from_utf8_lossy(&bytes),
                    count
                );
            }

            self.vocab.insert(new_id, bytes);
            self.ranks.insert(pair, i);
            self.merges.push(pair);
        }
    }

    /// Applies the learned merges in rank order.
    ///
    /// Important: not "most frequent merge first", but always the lowest rank
    /// still present. Only that makes encode() consistent with training.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids: Vec<u32> = text.bytes().map(u32::from).collect();

        while ids.len() >= 2 {
            let best = ids
                .windows(2)
                .filter_map(|w| {
                    let pair = (w[0], w[1]);
                    self.ranks.get(&pair).map(|&r| (r, pair))
                })
                .min_by_key(|&(r, _)| r);

            let Some((rank, pair)) = best else { break };
            ids = merge(&ids, pair, 256 + rank);
        }
        ids
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for id in ids {
            match self.vocab.get(id) {
                Some(b) => bytes.extend_from_slice(b),
                None => bytes.extend_from_slice("\u{FFFD}".as_bytes()),
            }
        }
        // lossy, because a partial token can end mid UTF-8 sequence
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Serialises only the merges — the rest is reconstructible from them.
    pub fn to_string_repr(&self) -> String {
        let mut s = String::from("mini-bpe v1\n");
        for (a, b) in &self.merges {
            s.push_str(&format!("{a} {b}\n"));
        }
        s
    }

    pub fn from_string_repr(s: &str) -> Result<Self, String> {
        let mut lines = s.lines();
        match lines.next() {
            Some("mini-bpe v1") => {}
            other => return Err(format!("unbekannter Header: {other:?}")),
        }

        let mut bpe = Bpe::new();
        for (i, line) in lines.filter(|l| !l.trim().is_empty()).enumerate() {
            let mut parts = line.split_whitespace();
            let a: u32 = parts.next().ok_or("Paar unvollständig")?
                .parse().map_err(|e| format!("{e}"))?;
            let b: u32 = parts.next().ok_or("Paar unvollständig")?
                .parse().map_err(|e| format!("{e}"))?;

            let new_id = 256 + i as u32;
            let mut bytes = bpe.vocab.get(&a).ok_or("unbekannte ID")?.clone();
            bytes.extend_from_slice(bpe.vocab.get(&b).ok_or("unbekannte ID")?);

            bpe.vocab.insert(new_id, bytes);
            bpe.ranks.insert((a, b), i as u32);
            bpe.merges.push((a, b));
        }
        Ok(bpe)
    }
}

fn count_pairs(ids: &[u32]) -> HashMap<(u32, u32), usize> {
    let mut stats = HashMap::new();
    for w in ids.windows(2) {
        *stats.entry((w[0], w[1])).or_insert(0) += 1;
    }
    stats
}

fn merge(ids: &[u32], pair: (u32, u32), new_id: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(ids.len());
    let mut i = 0;
    while i < ids.len() {
        if i + 1 < ids.len() && ids[i] == pair.0 && ids[i + 1] == pair.1 {
            out.push(new_id);
            i += 2;
        } else {
            out.push(ids[i]);
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_ohne_training() {
        let bpe = Bpe::new();
        let text = "Grüezi, Zürich! 🇨🇭";
        assert_eq!(bpe.decode(&bpe.encode(text)), text);
    }

    #[test]
    fn roundtrip_mit_training() {
        let corpus = "der Hund und die Katze und der Vogel und die Maus ".repeat(50);
        let mut bpe = Bpe::new();
        bpe.train(&corpus, 60, false);

        let text = "der Hund und die Katze";
        let ids = bpe.encode(text);

        assert_eq!(bpe.decode(&ids), text);
        // compression must be measurable, otherwise encode() is wrong
        assert!(ids.len() < text.len(), "{} vs {}", ids.len(), text.len());
    }

    #[test]
    fn serialisierung_ist_stabil() {
        let corpus = "abababab abcabcabc ".repeat(30);
        let mut a = Bpe::new();
        a.train(&corpus, 20, false);
        let b = Bpe::from_string_repr(&a.to_string_repr()).unwrap();
        assert_eq!(a.encode("abcabc"), b.encode("abcabc"));
    }
}
