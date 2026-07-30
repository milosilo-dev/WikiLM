use std::io::stdin;
use std::collections::{HashMap, HashSet};
use std::fs;

use tch::{Device, CModule, Tensor, Kind};

use crate::tokens::tokenise_text;

fn find_key_for_value(map: &HashMap<String, usize>, value: usize) -> Option<&String> {
    map.iter()
        .find_map(|(key, &val)| if val == value { Some(key) } else { None })
}

fn apply_repetition_penalty(logits: &mut Tensor, tokens: &[i32], penalty: f64) {
    let seen: HashSet<i32> = tokens.iter().cloned().collect();

    for &token_id in &seen {
        let idx = token_id as i64;
        let val = logits.double_value(&[idx]);
        let new_val = if val > 0.0 { val / penalty } else { val * penalty };
        // .get() returns a view into the same storage, so fill_ mutates logits in place
        let _ = logits.get(idx).fill(new_val);
    }
}

fn generate_next_token(
    model: &CModule,
    tokens: &[i32],
    device: Device,
    temperature: f64,
    top_k: i64,
    penalty: f64,
) -> (i64, i64) {
    let start = tokens.len().saturating_sub(64);
    let window = &tokens[start..];

    let input_tensor = Tensor::from_slice(window)
        .to_kind(Kind::Int64)
        .reshape([1, 64])
        .to_device(device);

    let output = model.forward_ts(&[input_tensor]).unwrap();

    // output: [1, 64, vocab] -> select last position -> [1, vocab] -> drop batch dim -> [vocab]
    let mut logits = output.select(1, 63).get(0);

    apply_repetition_penalty(&mut logits, tokens, penalty);

    let logits = logits / temperature;
    let log_probs = logits.log_softmax(-1, Kind::Float);

    let (values, indices) = logits.topk(top_k, -1, true, true);
    let probs = values.softmax(-1, Kind::Float);

    let choice = probs.multinomial(1, false); // shape [1]
    let choice_idx = choice.int64_value(&[0]);

    (indices.int64_value(&[choice_idx]), log_probs.int64_value(&[choice_idx]))
}

fn generate_beams(
    model: &CModule,
    prompt: &[i32],
    device: Device,
    amount: usize,
    beam_width: usize,
    top_k: i64,
) -> Vec<i32> {
    let mut beams: Vec<(Vec<i32>, f64)> = vec![(prompt.to_vec(), 0.0); beam_width];

    for beam in &mut beams {
        for _ in 0..amount{
            let (id, prob) = generate_next_token(&model, &beam.0.to_vec(), device, 0.8, top_k, 1.3);
            beam.0.push(id as i32);
            beam.1 += (prob as f64).log10()
        }
    }

    beams.iter().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0.to_vec()
}

pub fn run() {
    let device = Device::cuda_if_available();

    let model = CModule::load_on_device(
        "model.pt",
        device
    ).unwrap();

    let mut token_cache: HashMap<String, Vec<usize>> = HashMap::new();

    let data = fs::read_to_string("training_data/tokens.json").expect("Could not open 'training_data/tokens.json'.");
    let tokens_vec: Vec<String> = serde_json::from_str(&data).expect("Incorrect json formatting.");
    let token_map: HashMap<String, usize> = tokens_vec
        .into_iter()
        .enumerate()
        .map(|(index, item)| (item, index))
        .collect();

    println!(">");
    let mut input: String = String::new();
    stdin().read_line(&mut input).unwrap();
    while input != "quit" {
        let mut tokens: Vec<i32> = vec![];
        tokens.extend(tokenise_text(input.as_str(), &token_map, &mut token_cache).into_iter().map(|x| x as i32));

        while tokens.len() < 64 {
            tokens.insert(0, 0);
        }

        let output_vec: Vec<i32> = generate_beams(&model, &tokens, device, 30, 10, 40);
        
        for output in output_vec{
            println!("{}", find_key_for_value(&token_map, output as usize).unwrap());
        }

        println!(">");
        input = String::new();
        stdin().read_line(&mut input).unwrap();
    }
}