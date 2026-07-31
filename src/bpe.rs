use serde::Serialize;
use serde_json::Value;
use std::{
    collections::HashMap,
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
};

fn vector_to_pairs<T: Eq + Copy>(list: &Vec<T>) -> Vec<(T, T)> {
    let mut ret: Vec<(T, T)> = vec![];
    for item in 0..list.len() - 1 {
        ret.push((list[item], list[item + 1]))
    }
    ret
}

pub fn tokenise_string(
    input_string: &String,
    tokenset: &Vec<String>,
    tokenhash: &HashMap<String, usize>,
) -> Vec<usize> {
    let split_length = tokenset
        .iter()
        .max_by(|x, y| x.len().cmp(&y.len()))
        .unwrap()
        .len();
    let mut tokenised: Vec<usize> = vec![];
    for char in input_string.chars() {
        if let Some(&token_id) = tokenhash.get(&char.to_string()) {
            tokenised.push(token_id);
        }
    }
    for _ in 0..split_length {
        let mut merged = Vec::with_capacity(tokenised.len());
        let mut i = 0;
        while i < tokenised.len() {
            if i + 1 < tokenised.len() {
                let left = tokenised[i];
                let right = tokenised[i + 1];
                let mut s = String::new();
                s.push_str(&tokenset[left]);
                s.push_str(&tokenset[right]);
                if let Some(&id) = tokenhash.get(&s) {
                    merged.push(id);
                    i += 2;
                    continue;
                }
            }
            merged.push(tokenised[i]);
            i += 1;
        }
        tokenised = merged;
        if tokenised.len() < 3 {
            break;
        }
    }
    tokenised
}

const TARGET_VOCAB_SIZE: usize = 1000;
const INPUT_FILE_LENGTH: usize = 100000;

const INPUT_FILE: &str = "training_data/input.txt";
const OUTPUT_FILE: &str = "training_data/tokens.json";
const CHARSET_FILE: &str = "training_data/chars.json";
const MODEL_INFO_FILE: &str = "training_data/model.json";

fn read_charset(tokenset: &mut Vec<String>, tokenhash: &mut HashMap<String, usize>) {
    let chars_file_content = fs::read_to_string(CHARSET_FILE)
        .expect(format!("Failed to open '{}'.", CHARSET_FILE).as_str());
    let chars_file_parsed: Value = serde_json::from_str(&chars_file_content)
        .expect(format!("Failed to phase '{}'.", CHARSET_FILE).as_str());
    if let Some(obj) = chars_file_parsed.as_object() {
        for key in obj.keys() {
            let key = key.to_lowercase();
            if !tokenset.contains(&key) {
                tokenset.push(key);
            }
        }
    }
    for (i, token) in tokenset.iter().enumerate() {
        tokenhash.insert(token.clone(), i);
    }
}

fn read_input_file() -> String {
    let file = File::open(INPUT_FILE).unwrap();
    let reader = BufReader::new(file);
    let mut input_file: String = String::new();
    reader
        .take(INPUT_FILE_LENGTH as u64)
        .read_to_string(&mut input_file)
        .unwrap();
    input_file
}

fn write_tokenset(tokenset: &Vec<String>) {
    let output_file =
        File::create(OUTPUT_FILE).expect(&format!("Could not open '{}'.", OUTPUT_FILE).to_string());
    let mut writer = BufWriter::new(output_file);
    serde_json::to_writer(&mut writer, tokenset)
        .expect(&format!("Could not write '{}'.", OUTPUT_FILE).to_string());
    writer.flush().expect("Could not write to disk!");
    #[derive(Serialize)]
    struct ModelInfo {
        vocab_size: usize,
    }
    let info = ModelInfo {
        vocab_size: tokenset.len(),
    };
    let output_file = File::create(MODEL_INFO_FILE)
        .expect(format!("Could not open '{}'.", MODEL_INFO_FILE).as_str());
    let mut writer = BufWriter::new(output_file);
    serde_json::to_writer(&mut writer, &info).expect("Could not write output!");
    writer.flush().expect("Could not write to disk!");
}

fn most_common_pair(pairs: &Vec<(usize, usize)>) -> Option<(usize, usize)> {
    let mut counts = HashMap::new();
    for pair in pairs {
        *counts.entry(*pair).or_insert(0) += 1;
    }
    counts
        .iter()
        .max_by_key(|(_, count)| *count)
        .map(|(pair, _)| pair)
        .copied()
}

pub fn make_bpe_tokenset() {
    let mut tokenset: Vec<String> = vec![];
    let mut tokenhash: HashMap<String, usize> = HashMap::new();
    read_charset(&mut tokenset, &mut tokenhash);
    let input_file: String = read_input_file();
    while tokenset.len() <= TARGET_VOCAB_SIZE {
        println!("{}", tokenset.len());
        let tokenised_input_file = tokenise_string(&input_file, &mut tokenset, &mut tokenhash);
        let tokenised_paired_input = vector_to_pairs(&tokenised_input_file);
        let new_token = most_common_pair(&tokenised_paired_input).unwrap();
        println!("Token pair: {:?}", new_token);
        let mut s = String::new();
        s.push_str(&tokenset[new_token.0]);
        s.push_str(&tokenset[new_token.1]);
        tokenset.push(s.clone());
        tokenhash.insert(s, tokenset.len() - 1);
    }
    write_tokenset(&tokenset);
}
