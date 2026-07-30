use std::env;
use custom_llm::frontend::run;
use custom_llm::tokens::{token_list, tokenise_input, test_tokeniser};

fn main() {
    let args: Vec<String> = env::args().collect();

    match args[1].as_str(){
        "token-list" => token_list(),
        "tokenize-input" => tokenise_input(),
        "all" => {
            token_list();
            tokenise_input();
        }
        "run" => run(),
        "test" => test_tokeniser(),
        _ => {},
    }
}
