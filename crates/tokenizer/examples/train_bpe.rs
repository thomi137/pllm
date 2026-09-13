// Trains a BPE-Tokenizer on a Corpus and writes to a merge table.
//
//   cargo run --release -p pllm-tokenizer --example train_bpe -- \
//       data/deu_news_2024_100K-sentences.txt 500 out/bpe-de.txt
//
// Arguments: <corpus> [num_merges=500] [output=out/bpe.txt]
use pllm_tokenizer::Bpe;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

pub fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: train_bpe <corpus.txt> [num_merges] [output.txt]");
        return ExitCode::FAILURE;
    }

    let corpus_path = &args[1];
    let num_merges: u32 = args.get(2).map_or(500, |s| s.parse().expect("num_merges"));
    let out_path = args.get(3).map_or("out/bpe.txt", |s| s.as_str());

    let raw = match fs::read_to_string(corpus_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Korpus '{corpus_path}' nicht lesbar: {e}");
            return ExitCode::FAILURE;
        }
    };
    let corpus = clean(&raw);

    println!("Corpus : {corpus_path}");
    println!("Bytes  : {}", corpus.len());
    println!("Merges : {num_merges}\n");

    let t0 = std::time::Instant::now();
    let mut bpe = Bpe::new();
    bpe.train(&corpus, num_merges, true);
    println!("\ntraining: {:.1}s", t0.elapsed().as_secs_f32());
    println!("vocab: {} Tokens", bpe.vocab_size());

    // This is just a sanity check to measure compression.
    // Useful when ocmparing corpora of different languages.
    let sample: String = corpus.chars().take(200_000).collect();
    let ids = bpe.encode(&sample);
    let ratio = sample.len() as f32 / ids.len() as f32;
    println!("Compression: {ratio:.2} bytes/token");

    assert_eq!(bpe.decode(&ids), sample, "round trip from encode to decode not working!");
    println!("round-trip: ok");

    // Short Demo in German (allos compound words).
    let demo = "Der Tokenizer zerlegt Donaudampfschifffahrt in Zürich.";
    println!("\nDemo: {demo}");
    print!("  ");
    for id in bpe.encode(demo) {
        print!("[{}]", bpe.token_repr(id));
    }
    println!("\n  {} Tokens for {} Bytes", bpe.encode(demo).len(), demo.len());

    // The longest tokens say what the corpus repeats most.
    println!("\nLongest tokens:");
    for (id, text) in bpe.longest_tokens(15) {
        println!("  {id:>6}  {text:?}");
    }

    // Save
    if let Some(dir) = Path::new(out_path).parent() {
        fs::create_dir_all(dir).expect("Output-Verzeichnis");
    }
    fs::write(out_path, bpe.to_string_repr()).expect("schreiben");
    println!("\nGeschrieben: {out_path}");

    // Same table, human side: ids only in the file above, text here.
    let readable_path = format!("{out_path}.readable.txt");
    fs::write(&readable_path, bpe.merge_table()).expect("schreiben");
    println!("Geschrieben: {readable_path}");

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
