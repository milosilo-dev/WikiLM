import torch
import torch.nn.functional as F
import json

DEVICE = "cuda"

with open("training_data/tokens.json") as f:
    vocab = json.load(f)

token_map = {t:i for i,t in enumerate(vocab)}
token_to_word = {i:t for i,t in enumerate(vocab)}


# Same tokenizer logic as Rust
PREFIXES = [
    "un",
    "de",
    "re",
    "pre",
    "mis",
    "over",
    "under",
    "out",
    "be"
]


def tokenize_word(word):
    word = word.lower()

    # whole word token
    if word in token_map:
        return [token_map[word]]

    tokens = []

    # prefix
    for prefix in PREFIXES:
        if word.startswith(prefix):
            tokens.append(token_map[prefix])
            word = word[len(prefix):]
            break


    chars = list(word)

    i = 0

    while i < len(chars):

        # 3 char token
        if i + 3 <= len(chars):
            part = "".join(chars[i:i+3])

            if part in token_map:
                tokens.append(token_map[part])
                i += 3
                continue


        # 2 char token
        if i + 2 <= len(chars):
            part = "".join(chars[i:i+2])

            if part in token_map:
                tokens.append(token_map[part])
                i += 2
                continue


        # character
        ch = chars[i]

        if ch in token_map:
            tokens.append(token_map[ch])

        i += 1


    return tokens



def tokenize(text):
    output = []

    for word in text.split():
        output.extend(tokenize_word(word))

        # add whitespace token
        output.append(token_map[" "])

    return output



model = torch.jit.load(
    "model.pt",
    map_location=DEVICE
)

model.eval()


def generate(tokens, amount=20):

    tokens = tokens.copy()

    for _ in range(amount):

        x = torch.tensor(
            [tokens[-64:]],
            dtype=torch.long,
            device=DEVICE
        )

        with torch.no_grad():
            output = model(x)

        # take prediction for the final token
        logits = output[:, -1, :]

        # remove batch dimension
        logits = logits.squeeze(0)

        # temperature
        logits = logits / 0.8

        values, indices = torch.topk(logits, 40)

        probs = F.softmax(values / 0.8, dim=-1)

        choice = torch.multinomial(probs, 1).item()
        next_token = indices[choice].item()

        tokens.append(next_token)

        print(
            token_to_word[next_token],
            end="",
            flush=True
        )


    print()



while True:

    text = input("\n> ")

    if text == "quit":
        break


    tokens = tokenize(text)

    print("Prediction:")
    generate(tokens, 500)