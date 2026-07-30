use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::collections::HashMap;
use std::fs;

use serde_json::Value;
use serde::Serialize;

const TOKEN_GEN_LENGTH: u32 = 10000;
const PREFIXES: [&str; 9] = [
    " un", 
    " de", 
    " re", 
    " pre",
    " mis",
    " over",
    " under",
    " out", 
    " be"
];

#[derive(Serialize)]
struct ModelInfo{
    vocab_size: usize,
}

impl ModelInfo {
    pub fn new(vocab_size: usize) -> Self{
        Self { vocab_size }
    }

    pub fn export(&self, path: &str) {
        println!("Writing to '{}'.", path);
        let output_file = File::create(path).expect(format!("Could not open '{}'.", path).as_str());
        let mut writer = BufWriter::new(output_file);
        serde_json::to_writer(&mut writer, &self).expect("Could not write output!");
        writer.flush().expect("Could not write to disk!");
    }
}

pub fn token_list(){
    let input_file = File::open("training_data/input.txt")
        .expect("Could not open input file 'training_data/input.txt'.");
    let reader = BufReader::new(input_file);

    let mut words: Vec<String> = vec![];
    let mut lines_used = 0;
    for line in reader.lines() {
        lines_used += 1;

        let mut line = line.expect(format!("Line {} could not be read from input file.", lines_used).as_str());
        // line.retain(|c| !c.is_ascii_punctuation());
        line.retain(|c| c.is_ascii());
        line.truncate(line.trim_end().len());
        let line = line.to_lowercase();

        let line_words: Vec<&str> = line.split_whitespace().collect();
        let line_words: Vec<String> = line_words.into_iter().map(|s| format!(" {}", s)).collect();
        words.extend(line_words);

        if lines_used == TOKEN_GEN_LENGTH{
            break;
        }
    }
    
    let mut word_freq: HashMap<String, u32> = HashMap::new();
    for word in words {
        // let is_numeric = word.chars().all(|c| c.is_ascii_digit());
        if word != ""{
            if word_freq.contains_key(&word) {
                word_freq.insert(word.clone(), word_freq[&word.clone()] + 1);
            } else {
                word_freq.insert(word.clone(), 1);
            }
        }
    }

    let mut freq_vec: Vec<(&String, &u32)> = word_freq.iter().collect();
    freq_vec.sort_by(|a, b| b.1.cmp(a.1));
    let mut tokens = vec![" ".to_string()];
    tokens.extend(PREFIXES.iter().map(|s| s.to_string()));

    let chars_file_content = fs::read_to_string("training_data/chars.json")
        .expect("Failed to open 'training_data/chars.json'.");
    let chars_file_parsed: Value = serde_json::from_str(&chars_file_content)
        .expect("Failed to phase 'training_data/chars.json'.");
    if let Some(obj) = chars_file_parsed.as_object() {
        for key in obj.keys() {
            let key = key.to_lowercase();
            tokens.push(key);
        }
    }

    for i in 0..(freq_vec.len() as f32 * 0.25) as usize{
        let word = freq_vec[i].0;
        if !tokens.contains(word){
            tokens.push(word.clone());
        }
    }

    for i in (freq_vec.len() as f32 * 0.25) as usize..(freq_vec.len() as f32 * 0.75) as usize{
        let word = freq_vec[i].0;
        for i in (0..word.len()).step_by(3){
            if i + 3 <= word.len(){
                let section = &word[i..i+3].to_string();
                if !tokens.contains(section) && section.len() == 3{
                    tokens.push(section.clone());
                }
            }
        }

        for i in (0..word.len()).step_by(2){
            if i + 2 <= word.len(){
                let section = &word[i..i+2].to_string();
                if !tokens.contains(section) && section.len() == 2{
                    tokens.push(section.clone());
                }
            }
        }
    }

    let output_file = File::create("training_data/tokens.json").expect("Could not open 'training_data/tokens.json'.");
    let mut writer = BufWriter::new(output_file);
    serde_json::to_writer(&mut writer, &tokens).expect("Could not write output!");
    writer.flush().expect("Could not write to disk!");

    let modle_info = ModelInfo::new(tokens.len());
    modle_info.export("training_data/model.json");
}

pub fn tokenise_text(word: &str, token_map: &HashMap<String, usize>, token_cache: &mut HashMap<String, Vec<usize>>) -> Vec<usize>{
    let word = &word.to_lowercase();
    if let Some(&token) = token_map.get(word.as_str()){
        return vec![token]
    } else if let Some(token) = token_cache.get(word.as_str()) {
        return token.clone()
    }
    
    let mut tokenised_phrase: Vec<usize> = vec![];
    let mut l_word = word.clone();
    for prefix in PREFIXES{
        if word.starts_with(prefix){
            tokenised_phrase.push(token_map[prefix]);
            l_word = word.strip_prefix(prefix).unwrap().to_string();
            break;
        }
    }

    let mut i = 0;
    let chars: Vec<char> = l_word.chars().collect();
    while i < chars.len(){

        // Try a three charcter token
        if i +  3 <= chars.len(){
            let section = chars[i..i + 3].iter().collect::<String>();
            if let Some(token) = token_map.get(section.as_str()) {
                tokenised_phrase.push(*token);
                i += 3;
                continue;
            }
        }

        // Try a two charcter token
        if i + 2 <= chars.len(){
            let section = chars[i..i + 2].iter().collect::<String>();
            if let Some(token) = token_map.get(section.as_str()) {
                tokenised_phrase.push(*token);
                i += 2;
                continue;
            }
        }

        // Fallback to single character
        let ch = chars[i..i + 1].iter().collect::<String>();
        if let Some(token) = token_map.get(ch.as_str()){
            tokenised_phrase.push(*token);
        }

        i += 1;
    }

    token_cache.insert(word.to_string(), tokenised_phrase.clone());
    return tokenised_phrase;
}

pub fn tokenise_input(){
    let mut token_cache: HashMap<String, Vec<usize>> = HashMap::new();

    let data = fs::read_to_string("training_data/tokens.json").expect("Could not open 'training_data/tokens.json'.");
    let tokens_vec: Vec<String> = serde_json::from_str(&data).expect("Incorrect json formatting.");
    let token_map: HashMap<String, usize> = tokens_vec
        .into_iter()
        .enumerate()
        .map(|(index, item)| (item, index))
        .collect();

    let input_data = fs::read_to_string("training_data/input.txt").expect("Could not open input.txt");
    let output_tokens: Vec<usize> = tokenise_text(input_data.as_str(), &token_map, &mut token_cache);

    println!("Writing to output file!");
    let output_file = File::create("training_data/tokenized_input.json").expect("Could not open 'training_data/tokenized_input.json'.");
    let mut writer = BufWriter::new(output_file);
    serde_json::to_writer(&mut writer, &output_tokens).expect("Could not write output!");
    writer.flush().expect("Could not write to disk!");
}

pub fn test_tokeniser() {
    let mut token_cache: HashMap<String, Vec<usize>> = HashMap::new();

    let data = fs::read_to_string("training_data/tokens.json").expect("Could not open 'training_data/tokens.json'.");
    let tokens_vec: Vec<String> = serde_json::from_str(&data).expect("Incorrect json formatting.");
    let token_map: HashMap<String, usize> = tokens_vec
        .clone()
        .into_iter()
        .enumerate()
        .map(|(index, item)| (item, index))
        .collect();

    let tokens = tokenise_text("I love cheese!", &token_map, &mut token_cache);
    for token in tokens {
        print!("{}", tokens_vec[token]);
    }
    println!("");
}