use std::{collections::HashMap, fs::{self, File}, io::{BufReader, BufWriter, Read, Write}};

use recursive::recursive;
use serde::Serialize;
use serde_json::Value;

/*
    ["H", "HE", "HA", "HAHA"] -> Longest token first, always at the end.
    First, split the phase in half.
    Second, check both halves, are they in the token list, if so return the token.
    if one half is not, re run self witrh just that
    use result as token list, return it
*/

#[recursive]
pub fn tokenise_string(input_string: &String, tokenset: &mut Vec<String>, token_cache: Option<&mut HashMap<String, Vec<usize>>>, depth: u32) -> Vec<usize> {
    println!("Depth: {}, Length: {}", depth, input_string.len());

    if input_string.len() == 0 {
        return vec![]
    } else if let Some(token_id) = tokenset.iter().position(|x| x == input_string){
        return vec![token_id]
    } else if input_string.len() == 1{
        return vec![]
    } else if let Some(token_cache) = token_cache{
        if let Some(token_ids) = token_cache.get(input_string){
            return token_ids.clone();
        }

        let (first_half, second_half) = input_string.split_at(input_string.len() / 2);
        let ret = [tokenise_string(&first_half.to_string(), tokenset, Some(token_cache), depth + 1), tokenise_string(&second_half.to_string(), tokenset, Some(token_cache), depth + 1)]
            .concat();
        token_cache.insert(input_string.clone(), ret.clone());
        return ret;
    }

    let (first_half, second_half) = input_string.split_at(input_string.len() / 2);
    [tokenise_string(&first_half.to_string(), tokenset, None, depth + 1), tokenise_string(&second_half.to_string(), tokenset, None, depth + 1)]
        .concat()
}

/*
    First, tokenise with char level charcters
    Next, loop over sequence and find the most common token pair
    Create new token with the pair, replace all versions of it
    Continue untill the target vocab size has been met
*/

// Token list related constants
const TARGET_VOCAB_SIZE: usize = 50000; // Size of vocab that we result in end of iteration
const INPUT_FILE_LENGTH: usize = 100; // Number of charcters to read from the input file

const INPUT_FILE: &str = "training_data/input.txt";
const OUTPUT_FILE: &str = "training_data/output.json";
const CHARSET_FILE: &str = "training_data/chars.json";
const MODEL_INFO_FILE: &str = "training_data/model.json";

fn read_charset(tokenset: &mut Vec<String>) {
    let chars_file_content = fs::read_to_string(CHARSET_FILE)
        .expect(format!("Failed to open '{}'.", CHARSET_FILE).as_str());
    let chars_file_parsed: Value = serde_json::from_str(&chars_file_content)
        .expect(format!("Failed to phase '{}'.", CHARSET_FILE).as_str());
    if let Some(obj) = chars_file_parsed.as_object() {
        for key in obj.keys() {
            let key = key.to_lowercase();
            tokenset.push(key);
        }
    }
}

fn read_input_file() -> String {
    let file = File::open(INPUT_FILE).unwrap();
    let reader = BufReader::new(file);
    let mut input_file: String = String::new();
    reader.take(INPUT_FILE_LENGTH as u64).read_to_string(&mut input_file).unwrap();
    input_file
}

fn write_tokenset(tokenset: &Vec<String>) {
    let output_file = File::create(OUTPUT_FILE).expect(&format!("Could not open '{}'.", OUTPUT_FILE).to_string());
    let mut writer = BufWriter::new(output_file);
    serde_json::to_writer(&mut writer, tokenset).expect(&format!("Could not write '{}'.", OUTPUT_FILE).to_string());
    writer.flush().expect("Could not write to disk!");

    #[derive(Serialize)]
    struct ModelInfo{vocab_size: usize}
    let info = ModelInfo{vocab_size: tokenset.len()};

    let output_file = File::create(MODEL_INFO_FILE).expect(format!("Could not open '{}'.", MODEL_INFO_FILE).as_str());
    let mut writer = BufWriter::new(output_file);
    serde_json::to_writer(&mut writer, &info).expect("Could not write output!");
    writer.flush().expect("Could not write to disk!");
}

fn vector_to_pairs<T: Eq + Copy>(list: Vec<T>) -> Vec<(T, T)> {
    let mut ret: Vec<(T, T)> = vec![];
    for item in 0..list.len() - 1 {
        ret.push((list[item], list[item + 1]))
    }
    ret
}

fn most_common_pair(pairs: &Vec<(usize, usize)>) -> Option<(usize, usize)> {
    let mut counts = HashMap::new();

    for pair in pairs {
        *counts.entry(*pair).or_insert(0) += 1;
    }

    counts.iter()
        .max_by_key(|(_, count)| *count)
        .map(|(pair, _)| pair)
        .copied()
}

pub fn make_bpe_tokenset() {
    let mut tokenset: Vec<String> = vec![];
    read_charset(&mut tokenset);

    let input_file: String = read_input_file();
    let mut token_cache: HashMap<String, Vec<usize>> = HashMap::new();

    while tokenset.len() != TARGET_VOCAB_SIZE {
        println!("{}", tokenset.len());
        let tokenised_input_file = tokenise_string(&input_file, &mut tokenset, Some(&mut token_cache), 0);
        let tokenised_paired_input = vector_to_pairs(tokenised_input_file);

        let new_token = most_common_pair(&tokenised_paired_input).unwrap();
        tokenset.push(format!("{}{}", tokenset[new_token.0], tokenset[new_token.1]))
    }

    write_tokenset(&tokenset);
}