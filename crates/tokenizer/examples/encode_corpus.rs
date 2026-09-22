// Trains a BPE tokenizer on the whole corpus and saves both the merge table
// and a fixed-size tokenized corpus, for reuse by the `model` crate.
//
//   cargo run --release -p pllm-tokenizer --example encode_corpus -- \
//       data/deu-ch_newscrawl_2012_1M-sentences.txt 500 1000000 out/corpus
//
// Arguments: <corpus> [num_merges=500] [num_tokens=1000000] [out_prefix=out/corpus]
// (same argument order as train_bpe.rs: corpus, merges, then the rest)
//
// Writes:
//   <out_prefix>.bpe               merge table, reusable (Bpe::from_string_repr)
//   <out_prefix>.bpe.readable.txt  merge table, human-readable
//   <out_prefix>.ids               num_tokens token ids, binary (ids_to_bytes)
use pllm_tokenizer::{Bpe, ids_to_bytes};
use std::fs;
use std::path::Path;
use std::process::ExitCode;

pub fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: encode_corpus <corpus.txt> [num_merges] [num_tokens] [out_prefix]");
        return ExitCode::FAILURE;
    }

    let corpus_path = &args[1];
    let num_merges: u32 = args.get(2).map_or(500, |s| s.parse().expect("num_merges"));
    let num_tokens: usize = args.get(3).map_or(1_000_000, |s| s.parse().expect("num_tokens"));
    let out_prefix = args.get(4).map_or("out/corpus", |s| s.as_str());

    let raw = match fs::read_to_string(corpus_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Korpus '{corpus_path}' nicht lesbar: {e}");
            return ExitCode::FAILURE;
        }
    };
    let corpus = clean(&raw);

    println!("Corpus       : {corpus_path}");
    println!("Corpus bytes : {}", corpus.len());
    println!("Merges       : {num_merges}");
    println!("Target tokens: {num_tokens}\n");

    // train() is O(num_merges * corpus_len): on the full 115 MB file this
    // rescans everything on every merge, so it is minutes, not seconds.
    // verbose=true so the run shows it is alive rather than looking hung.
    let t0 = std::time::Instant::now();
    let mut bpe = Bpe::new();
    bpe.train(&corpus, num_merges, true);
    println!("\ntraining : {:.1}s, vocab {}", t0.elapsed().as_secs_f32(), bpe.vocab_size());

    let ids = bpe.encode(&corpus);
    let ratio = corpus.len() as f32 / ids.len() as f32;
    println!("encoding : {} tokens from {} bytes ({:.2} bytes/token)", ids.len(), corpus.len(), ratio);

    if ids.len() < num_tokens {
        eprintln!("\nnur {} Tokens erzeugt, {num_tokens} gefordert", ids.len());
        return ExitCode::FAILURE;
    }
    let ids = &ids[..num_tokens];

    if let Some(dir) = Path::new(out_prefix).parent() {
        fs::create_dir_all(dir).expect("Output-Verzeichnis");
    }

    let bpe_path = format!("{out_prefix}.bpe");
    fs::write(&bpe_path, bpe.to_string_repr()).expect("schreiben");
    println!("\nGeschrieben: {bpe_path}");

    let readable_path = format!("{out_prefix}.bpe.readable.txt");
    fs::write(&readable_path, bpe.merge_table()).expect("schreiben");
    println!("Geschrieben: {readable_path}");

    let ids_path = format!("{out_prefix}.ids");
    let ids_bytes = ids_to_bytes(ids).expect("ids passen in u16 (vocab_size < 65536)");
    fs::write(&ids_path, &ids_bytes).expect("schreiben");
    println!("Geschrieben: {ids_path}  ({} Tokens, {} Bytes)", ids.len(), ids_bytes.len());

    ExitCode::SUCCESS
}

fn clean(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for line in raw.lines() {
        let text = match line.split_once('\t') {
            Some((id, rest)) if id.chars().all(|c| c.is_ascii_digit()) => rest,
            _ => line,
        };
        let text = text.trim();
        if !text.is_empty() {
            out.push_str(text);
            out.push('\n');
        }
    }
    out
}
