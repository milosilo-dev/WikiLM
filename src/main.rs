use std::env;

fn token_list(){

}

fn tokenise_input(){
    
}

fn main() {
    let args: Vec<String> = env::args().collect();

    match args[1].as_str(){
        "token-list" => token_list(),
        "tokenize-input" => tokenise_input(),
        _ => {},
    }
}
