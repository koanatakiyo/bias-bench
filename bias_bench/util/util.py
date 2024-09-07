from transformers import AutoTokenizer

def _is_generative(model):
    # Checks if we are running an autoregressive model.
    return model in [
        "GPT2LMHeadModel",
        "SentenceDebiasGPT2LMHeadModel",
        "INLPGPT2LMHeadModel",
        "CDAGPT2LMHeadModel",
        "DropoutGPT2LMHeadModel",
        "SelfDebiasGPT2LMHeadModel",
    ]


def get_self_debias_prefix_token_count(model_name):
    DEBIASING_PREFIXES = {
        "race-color": "The following text discriminates against people because of their race/color: ",
        "gender": "The following text discriminates against people because of their gender: ",
        "socioeconomic": "The following text discriminates against people because of their socioeconomic status/occupation: ",
        "sexual-orientation": "The following text discriminates against people because of their sexual orientiation: ",
        "religion": "The following text discriminates against people because of their religion: ",
        "age": "The following text discriminates against people because of their age: ",
        "nationality": "The following text discriminates against people because of their nationality: ",
        "disability": "The following text discriminates against people because of their disability: ",
        "physical-appearance": "The following text discriminates against people because of their physical appearance: ",
    }
    # Encode the prefixes
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prefix_token_counts = {}

    for bias_type, prefix in DEBIASING_PREFIXES.items():
        input_ids = tokenizer.encode(prefix, return_tensors="pt")
        prefix_token_counts[bias_type] = len(input_ids[0])
        print(f"Model: {model_name}, Bias Type: {bias_type}, Prefix Token Count: {len(input_ids[0])}")

    return prefix_token_counts

def _is_self_debias(model):
    # Checks if we are running a Self-Debias model.
    return model in [
        "SelfDebiasGPT2LMHeadModel",
        "SelfDebiasBertForMaskedLM",
        "SelfDebiasAlbertForMaskedLM",
        "SelfDebiasRobertaForMaskedLM",

        "SelfDebiasPhi2LMHeadModel", # For phi 2 models
        "SelfDebiasLlama2LMHeadModel", # For llama 2 models
        "SelfDebiasMistrailLMHeadModel", # For mistrail 2 models

    ]
