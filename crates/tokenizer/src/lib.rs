//! This is the tokenizer. I follows a BPE Algorithm
//! See [Sennrich, Haddow & Birch (2016) — "Neural Machine Translation of Rare Words with Subword Units"](arxiv.org/abs/1508.07909 Sections 3, 4, 5 — Byte Pair Encoding (BPE))
use std::cmp::Reverse;
use std::collections::HashMap;

/// This is the tokenizer struct.
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
        // One byte sequence per chunk, never one for the whole corpus. Pairs
        // are only ever counted and merged inside a chunk, so no token can
        // grow across a word boundary.
        let mut chunks: Vec<Vec<u32>> = split_chunks(corpus)
            .into_iter()
            .map(|chunk| chunk.bytes().map(u32::from).collect())
            .collect();

        for i in 0..num_merges {
            let mut stats = HashMap::new();
            for ids in &chunks {
                count_pairs_into(ids, &mut stats);
            }
            // Tie-break on the pair itself, smallest first. HashMap iteration
            // order is random, so without this two runs on the same corpus
            // learn different merges whenever counts tie.
            let Some((&pair, &count)) = stats.iter().max_by_key(|&(&p, &c)| (c, Reverse(p))) else {
                break;
            };
            if count < 2 {
                break; // nothing left to gain
            }

            let new_id = 256 + i;
            for ids in &mut chunks {
                *ids = merge(ids, pair, new_id);
            }

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
        // Same split as in train(), or encode() would apply merges the
        // training never saw.
        let mut out = Vec::new();
        for chunk in split_chunks(text) {
            out.extend(self.encode_chunk(chunk));
        }
        out
    }

    fn encode_chunk(&self, chunk: &str) -> Vec<u32> {
        let mut ids: Vec<u32> = chunk.bytes().map(u32::from).collect();

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

    /// One token, rendered for human eyes.
    ///
    /// Whitespace is made visible, because where the spaces ended up is the
    /// point of looking at a merge table at all. A token can also stop
    /// mid-UTF-8 — the first byte of "ü" is a legal token — so an invalid
    /// sequence is shown as its raw bytes rather than as the replacement
    /// character `decode()` would produce, which hides which byte it was.
    pub fn token_repr(&self, id: u32) -> String {
        let Some(bytes) = self.vocab.get(&id) else {
            return format!("<unknown {id}>");
        };
        match std::str::from_utf8(bytes) {
            Ok(s) => s.replace(' ', "␣").replace('\n', "⏎").replace('\t', "⇥"),
            Err(_) => bytes.iter().map(|b| format!("<{b:02X}>")).collect(),
        }
    }

    /// The learned merges in rank order, one line each:
    ///
    /// ```text
    ///     0     256  "e" + "r" -> "er"
    ///     1     257  "␣d" + "er" -> "␣der"
    /// ```
    ///
    /// Reading this top to bottom is reading the corpus statistics: letter
    /// pairs first, then syllables, then whole frequent words with their
    /// leading space. This is the human counterpart to `to_string_repr`,
    /// which writes ids only so that it can be read back exactly.
    pub fn merge_table(&self) -> String {
        let mut out = String::new();
        for (rank, &(a, b)) in self.merges.iter().enumerate() {
            let new_id = 256 + rank as u32;
            out.push_str(&format!(
                "{rank:>5}  {new_id:>6}  {:?} + {:?} -> {:?}\n",
                self.token_repr(a),
                self.token_repr(b),
                self.token_repr(new_id)
            ));
        }
        out
    }

    /// The `n` longest learned tokens, longest first. Raw bytes are skipped.
    ///
    /// Length is the interesting axis: the longest tokens are whatever the
    /// corpus repeats most, which is where German compounds show up.
    pub fn longest_tokens(&self, n: usize) -> Vec<(u32, String)> {
        let mut learned: Vec<u32> = self.vocab.keys().copied().filter(|&id| id >= 256).collect();
        // Length first, then id — equal-length tokens keep a stable order
        // instead of inheriting the HashMap's random one.
        learned.sort_by_key(|&id| (Reverse(self.vocab[&id].len()), id));
        learned
            .into_iter()
            .take(n)
            .map(|id| (id, self.token_repr(id)))
            .collect()
    }

    /// Serialises only the merges — the rest is reconstructible from them.
    pub fn to_string_repr(&self) -> String {
        let mut s = String::from("pllm bpe v1\n");
        for (a, b) in &self.merges {
            s.push_str(&format!("{a} {b}\n"));
        }
        s
    }

    pub fn from_string_repr(s: &str) -> Result<Self, String> {
        let mut lines = s.lines();
        match lines.next() {
            Some("pllm bpe v1") => {}
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

/// Splits text into chunks that BPE may not merge across.
///
/// This is the hand-written stand-in for the GPT-2 pre-tokenizer regex. That
/// regex is just an enumeration of character classes, and `tokenizer` carries
/// no dependencies, so it is spelled out as a state machine here.
///
/// Four classes, each able to keep one leading space — which is why `" der"`
/// and `"der"` end up as different tokens:
///
/// 1. optional space + run of letters
/// 2. optional space + a *single* digit
/// 3. optional space + run of punctuation/symbols
/// 4. run of whitespace
///
/// Digits are emitted one at a time on purpose: otherwise `"1234"` is cut
/// into `"12"+"34"` or `"1"+"234"` depending on corpus frequency, and the
/// same number gets different tokens in different contexts.
fn char_at(text: &str, i: usize) -> Option<char> {
    text.get(i..).and_then(|s| s.chars().next())
}

/// Serialises token ids for reuse elsewhere (e.g. the `model` crate's
/// training loop), without re-running `encode()` every time.
///
/// Format: an ASCII header line `"pllm ids v1\n"`, then every id as a raw
/// little-endian `u16`. `u16` because a realistic vocab (a few hundred to a
/// few thousand merges) fits easily, and it halves the file versus `u32`.
/// Errors instead of silently truncating if a vocab ever grows past 65535.
pub fn ids_to_bytes(ids: &[u32]) -> Result<Vec<u8>, String> {
    let mut out = Vec::from(b"pllm ids v1\n".as_slice());
    for &id in ids {
        let id: u16 = id.try_into().map_err(|_| format!("id {id} does not fit in u16"))?;
        out.extend_from_slice(&id.to_le_bytes());
    }
    Ok(out)
}

pub fn ids_from_bytes(bytes: &[u8]) -> Result<Vec<u32>, String> {
    let header = b"pllm ids v1\n";
    let body = bytes.strip_prefix(header.as_slice()).ok_or("unbekannter Header")?;
    if body.len() % 2 != 0 {
        return Err(format!("{} ist keine gerade Anzahl Bytes", body.len()));
    }
    Ok(body.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as u32).collect())
}

pub fn split_chunks(text: &str) -> Vec<&str> {
    let mut chunks = Vec::new();
    let mut i = 0; // Byte-Index

    while i < text.len() {
        let start = i;
        let mut j = i;

        // A trailing whitespace belongs to the following chunk,
        // but only if there is a non-whitespace. ' ' is 1 byte, so
        // j + 1 is the position of the (beginning) of the next char.
        if char_at(text, j) == Some(' ')
            && matches!(char_at(text, j + 1), Some(c) if !c.is_whitespace()){
            j += 1;
        } 

        let c = char_at(text, j).expect("j < text.len()");

        if c.is_alphabetic() {
            while let Some(c) = char_at(text, j) {
                if !c.is_alphabetic() { break }
                j += c.len_utf8();
            }
        } else if c.is_numeric() {
            j += c.len_utf8(); // eine Ziffer pro Chunk
        } else if !c.is_whitespace() {
            while let Some(c) = char_at(text, j) {
                if c.is_whitespace() || c.is_alphanumeric() { break }
                j += c.len_utf8();
            }
        } else {
            let mut last_space = None;
            let mut n = 0;
            while let Some(c) = char_at(text, j) {
                if !c.is_whitespace() { break }
                if c == ' ' { last_space = Some(j); }
                j += c.len_utf8();
                n += 1;
            }
            // Letztes Leerzeichen an das folgende Wort zurückgeben,
            // aber nur bei einem Run von mindestens zwei Zeichen.
            if j < text.len() && n > 1 {
                if let Some(pos) = last_space {
                    if pos + 1 == j { j = pos; }
                }
            }
        }

        chunks.push(&text[start..j]);
        i = j;
    }
    chunks
}

fn count_pairs_into(ids: &[u32], stats: &mut HashMap<(u32, u32), usize>) {
    for w in ids.windows(2) {
        *stats.entry((w[0], w[1])).or_insert(0) += 1;
    }
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
    fn roundtrip_without_training() {
        let bpe = Bpe::new();
        let text = "Grüezi, Zürich! 🇨🇭";
        assert_eq!(bpe.decode(&bpe.encode(text)), text);
    }

    #[test]
    fn roundtrip_with_training() {
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
    fn chunks_keep_the_leading_space() {
        assert_eq!(split_chunks("der Hund"), vec!["der", " Hund"]);
        assert_eq!(split_chunks("Hallo, Welt!\n"), vec!["Hallo", ",", " Welt", "!", "\n"]);
        assert_eq!(split_chunks("  x"), vec![" ", " x"]);
    }

    #[test]
    fn no_merge_crosses_a_word_boundary() {
        // "der Hund" is the most frequent pair sequence here, so without the
        // pre-split BPE would happily learn it as one token.
        let corpus = "der Hund der Hund der Hund ".repeat(100);
        let mut bpe = Bpe::new();
        bpe.train(&corpus, 80, false);

        for (id, bytes) in &bpe.vocab {
            if *id < 256 {
                continue;
            }
            // A space is allowed only as the first byte of a token.
            assert!(
                !bytes[1..].contains(&b' '),
                "token {id} = {:?} spans a word boundary",
                String::from_utf8_lossy(bytes)
            );
        }
    }

    #[test]
    fn digits_stay_single_tokens() {
        let corpus = "1234 1234 1234 ".repeat(100);
        let mut bpe = Bpe::new();
        bpe.train(&corpus, 50, false);

        // Four digits, four tokens — no matter how often "1234" occurred.
        assert_eq!(bpe.encode("1234").len(), 4);
        assert_eq!(bpe.decode(&bpe.encode("1234")), "1234");
    }

    #[test]
    fn training_is_deterministic() {
        // Ties in pair counts are common once chunks are small; HashMap order
        // must not decide which merge is learned.
        let corpus = "ab cd ab cd ef ef ".repeat(40);
        let mut a = Bpe::new();
        let mut b = Bpe::new();
        a.train(&corpus, 30, false);
        b.train(&corpus, 30, false);
        assert_eq!(a.to_string_repr(), b.to_string_repr());
    }

    #[test]
    fn token_repr_makes_whitespace_visible() {
        let mut bpe = Bpe::new();
        bpe.train(&"der Hund der Hund der Hund ".repeat(50), 40, false);

        assert_eq!(bpe.token_repr(b' ' as u32), "␣");
        assert_eq!(bpe.token_repr(b'\n' as u32), "⏎");
        // First byte of "ü" — valid token, invalid UTF-8 on its own.
        assert_eq!(bpe.token_repr(0xC3), "<C3>");
        assert_eq!(bpe.token_repr(9_999), "<unknown 9999>");
    }

    #[test]
    fn merge_table_is_readable() {
        let corpus = "der Hund der Hund der Hund ".repeat(50);
        let mut bpe = Bpe::new();
        bpe.train(&corpus, 40, false);

        let table = bpe.merge_table();
        // One line per learned merge, ranks in order starting at 0.
        assert_eq!(table.lines().count(), bpe.vocab_size() - 256);
        assert!(table.lines().next().unwrap().starts_with("    0     256"));
        // The whole word must be in there, with its leading space marked.
        assert!(table.contains("\"␣Hund\""), "{table}");
    }

    #[test]
    fn longest_tokens_are_sorted_long_first() {
        let corpus = "Donaudampfschifffahrt Donaudampfschifffahrt ab ".repeat(50);
        let mut bpe = Bpe::new();
        bpe.train(&corpus, 60, false);

        let top = bpe.longest_tokens(5);
        assert_eq!(top.len(), 5);
        for pair in top.windows(2) {
            assert!(pair[0].1.chars().count() >= pair[1].1.chars().count(), "{top:?}");
        }
        assert!(top[0].1.contains("Donaudampfschifffahrt"), "{top:?}");
    }

    #[test]
    fn stable_serialization() {
        let corpus = "abababab abcabcabc ".repeat(30);
        let mut a = Bpe::new();
        a.train(&corpus, 20, false);
        let b = Bpe::from_string_repr(&a.to_string_repr()).unwrap();
        assert_eq!(a.encode("abcabc"), b.encode("abcabc"));
    }

    #[test]
    fn ids_roundtrip() {
        let ids = vec![0, 255, 256, 65535];
        let bytes = ids_to_bytes(&ids).unwrap();
        assert_eq!(ids_from_bytes(&bytes).unwrap(), ids);
    }

    #[test]
    fn ids_reject_out_of_range() {
        assert!(ids_to_bytes(&[65536]).is_err());
    }

    #[test]
    fn ids_reject_unknown_header() {
        assert!(ids_from_bytes(b"not the right header\n").is_err());
    }
}
