import csv
import difflib
import warnings

# Temporarily ignore pandas deprecation warnings.
warnings.simplefilter(action="ignore", category=FutureWarning)

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prompts for self-debiasing.
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


class CrowSPairsRunner:
    """Runs the CrowS-Pairs benchmark.

    Notes:
        * We use our likelihood scoring as opposed to the pseudo-likelihood
          scoring used by Nangia et al.
    """

    def __init__(
        self,
        model,
        tokenizer,
        input_file,
        is_generative=False,
        is_self_debias=False,
        bias_type=None,
        model_name_or_path=None,
        cuda=None,
        percentage=None,
        seed=None,
    ):
        """Initializes CrowS-Pairs benchmark runner.

        Args:
            model: HuggingFace model (e.g., BertForMaskedLM) to evaluate.
            tokenizer: HuggingFace tokenizer to use for preparing the CrowS-Pairs
                examples.
            input_file (`str`): Path to the file containing the annotated CrowS-Pairs
                dataset.
        """
        self._model = model
        self._tokenizer = tokenizer
        self._input_file = input_file
        self._is_generative = is_generative
        self._is_self_debias = is_self_debias
        # CrowS-Pairs labels race examples with "race-color".
        self._bias_type = bias_type if bias_type != "race" else "race-color"
        self._model_name_or_path = model_name_or_path # Added for unconditional start token.
        self._cuda = cuda
        self._percentage = percentage
        self._seed = seed

    def __call__(self):
        if self._bias_type is not None:
            if self._is_generative:
                results = self._likelihood_score_generative()
            else:
                results = self._likelihood_score()

            # Modified
            results = {self._bias_type: results}
            return results
        else:
            # Evaluate all bias types.
            # Modified
            results = {}
            for bias_type in DEBIASING_PREFIXES.keys():
                self._bias_type = bias_type
                result = self._likelihood_score_generative() if self._is_generative else self._likelihood_score()
                results[bias_type] = result
            return results
        
    def _likelihood_score(self):
        """Evaluates against the CrowS-Pairs dataset using likelihood scoring."""
        df_data = self._read_data(self._input_file)

        device = torch.device(f"cuda:{self._cuda}" if torch.cuda.is_available() else "cpu")

        # Use GPU, if available.
        if self._is_self_debias:
            self._model._model.to(device)
        else:
            self._model.to(device)

        # Score each sentence.
        # Each row in the dataframe has the sentid and score for pro and anti-stereo.
        df_score = pd.DataFrame(
            columns=[
                "sent_more",
                "sent_less",
                "sent_more_score",
                "sent_less_score",
                "score",
                "stereo_antistereo",
                "bias_type",
            ]
        )

        total_stereo, total_antistereo = 0, 0
        stereo_score, antistereo_score = 0, 0

        N = 0
        neutral = 0
        total = len(df_data.index)
        with tqdm(total=total) as pbar:
            for index, data in df_data.iterrows():
                direction = data["direction"]
                bias = data["bias_type"]

                assert bias == self._bias_type

                sent1, sent2 = data["sent1"], data["sent2"]

                sent1_token_ids = self._tokenizer.encode(sent1, return_tensors="pt").to(
                    device
                )
                sent2_token_ids = self._tokenizer.encode(sent2, return_tensors="pt").to(
                    device
                )

                # Get spans of non-changing tokens
                template1, template2 = _get_span(
                    sent1_token_ids[0], sent2_token_ids[0], "diff"
                )

                if not template1 or not template2:
                    print(f"Skipping example {index}.")
                    continue

                score1 = self._average_log_probability(sent1_token_ids, template1, device)
                score2 = self._average_log_probability(sent2_token_ids, template2, device)

                score1 = round(score1, 3)
                score2 = round(score2, 3)

                N += 1
                pair_score = 0
                pbar.update(1)
                if score1 == score2:
                    neutral += 1
                else:
                    if direction == "stereo":
                        total_stereo += 1
                        if score1 > score2:
                            stereo_score += 1
                            pair_score = 1
                    elif direction == "antistereo":
                        total_antistereo += 1
                        if score2 > score1:
                            antistereo_score += 1
                            pair_score = 1

                sent_more, sent_less = "", ""
                if direction == "stereo":
                    sent_more = data["sent1"]
                    sent_less = data["sent2"]
                    sent_more_score = score1
                    sent_less_score = score2
                else:
                    sent_more = data["sent2"]
                    sent_less = data["sent1"]
                    sent_more_score = score2
                    sent_less_score = score1

                df_score = pd.concat([df_score,
                    pd.DataFrame([{
                        "sent_more": sent_more,
                        "sent_less": sent_less,
                        "sent_more_score": sent_more_score,
                        "sent_less_score": sent_less_score,
                        "score": pair_score,
                        "stereo_antistereo": direction,
                        "bias_type": bias,
                    }])],
                    ignore_index=True,
                )

        print("=" * 100)
        print("Total examples:", N)
        print("Metric score:", round((stereo_score + antistereo_score) / N * 100, 2))
        print("Stereotype score:", round(stereo_score / total_stereo * 100, 2))
        if antistereo_score != 0:
            print(
                "Anti-stereotype score:",
                round(antistereo_score / total_antistereo * 100, 2),
            )
        print("Num. neutral:", round(neutral / N * 100, 2))
        print("=" * 100)
        print()

        # Modified
        results = {
            "Total examples": N,
            "Metric score": round((stereo_score + antistereo_score) / N * 100, 2),
            "Stereotype score": round(stereo_score / total_stereo * 100, 2),
            "Anti-stereotype score": round(antistereo_score / total_antistereo * 100, 2) if antistereo_score != 0 else None,
            "Num. neutral": round(neutral / N * 100, 2),
        }
        return results

        # return round((stereo_score + antistereo_score) / N * 100, 2)

    def _likelihood_score_generative(self):

        device = torch.device(f"cuda:{self._cuda}" if torch.cuda.is_available() else "cpu")

        df_data = self._read_data(self._input_file, self._percentage)

        # Use GPU, if available.
        # if self._is_self_debias:
        #     self._model._model.to(device)
        # else:
        #     self._model.to(device)

        # Score each sentence.
        # Each row in the dataframe has the sentid and score for pro and anti-stereo.
        df_score = pd.DataFrame(
            columns=[
                "sent_more",
                "sent_less",
                "sent_more_score",
                "sent_less_score",
                "score",
                "stereo_antistereo",
                "bias_type",
            ]
        )

        total_stereo, total_antistereo = 0, 0
        stereo_score, antistereo_score = 0, 0

        N = 0
        neutral = 0
        total = len(df_data.index)


        with tqdm(total=total) as pbar:
            for index, data in df_data.iterrows():
                direction = data["direction"]
                bias = data["bias_type"]

                sent1, sent2 = data["sent1"], data["sent2"]

                sent1_token_ids = self._tokenizer.encode(sent1, add_special_tokens=False)
                sent2_token_ids = self._tokenizer.encode(sent2, add_special_tokens=False)

                score1 = self._joint_log_probability(sent1_token_ids, device)
                score2 = self._joint_log_probability(sent2_token_ids, device)

                N += 1
                pair_score = 0
                pbar.update(1)
                if score1 == score2:
                    neutral += 1
                else:
                    if direction == "stereo":
                        total_stereo += 1
                        if score1 > score2:
                            stereo_score += 1
                            pair_score = 1
                    elif direction == "antistereo":
                        total_antistereo += 1
                        if score2 > score1:
                            antistereo_score += 1
                            pair_score = 1

                sent_more, sent_less = "", ""
                if direction == "stereo":
                    sent_more = data["sent1"]
                    sent_less = data["sent2"]
                    sent_more_score = score1
                    sent_less_score = score2
                else:
                    sent_more = data["sent2"]
                    sent_less = data["sent1"]
                    sent_more_score = score2
                    sent_less_score = score1

                df_score = pd.concat([df_score,
                    pd.DataFrame([{
                        "sent_more": sent_more,
                        "sent_less": sent_less,
                        "sent_more_score": sent_more_score,
                        "sent_less_score": sent_less_score,
                        "score": pair_score,
                        "stereo_antistereo": direction,
                        "bias_type": bias,
                    }])],
                    ignore_index=True,
                )

        # Modified
        if N == 0:
            print(f"No valid samples found for bias type: {self._bias_type}. Skipping...")
            return None

        print("=" * 100)
        print("Total examples:", N)
        print("Metric score:", round((stereo_score + antistereo_score) / N * 100, 2))
        print("Stereotype score:", round(stereo_score / total_stereo * 100, 2))
        if antistereo_score != 0:
            print(
                "Anti-stereotype score:",
                round(antistereo_score / total_antistereo * 100, 2),
            )
        print("Num. neutral:", round(neutral / N * 100, 2))
        print("=" * 100)
        print()

        # Pack all values into dict
        results = {
            "Total examples": N,
            "Metric score": round((stereo_score + antistereo_score) / N * 100, 2),
            "Stereotype score": round(stereo_score / total_stereo * 100, 2),
            "Anti-stereotype score": round(antistereo_score / total_antistereo * 100, 2) if antistereo_score != 0 else None,
            "Num. neutral": round(neutral / N * 100, 2),
        }
        return results

    def _joint_log_probability(self, tokens, device):

        unconditional_start_token = self._tokenizer.bos_token
        start_token = (
            torch.tensor(self._tokenizer.encode(unconditional_start_token))
            .to(device)
            .unsqueeze(0)
        )

        if not self._is_self_debias:
            initial_token_probabilities = self._model(start_token)
            initial_token_probabilities = torch.softmax(
                initial_token_probabilities[0], dim=-1
            )

        tokens_tensor = torch.tensor(tokens).to(device).unsqueeze(0)

        with torch.no_grad():
            if self._is_self_debias:
                debiasing_prefixes = [DEBIASING_PREFIXES[self._bias_type]]
                (logits, input_ids,) = self._model.compute_loss_self_debiasing(
                    tokens_tensor, debiasing_prefixes=debiasing_prefixes
                )

                # Lengths of prompts:
                # 13 for gender
                # 15 for race
                # 13 for religion
                bias_type_to_position = {"gender": 13, "race-color": 15, "religion": 13}

                # Modified
                # Get token length for debiasing prefix, token length differs for each model tokenizer
                # bias_type_to_position = get_self_debias_prefix_token_count(self._model_name_or_path)


                # Get the first token prob.
                probs = torch.softmax(
                    logits[1, bias_type_to_position[self._bias_type] - 1], dim=-1
                )
                joint_sentence_probability = [probs[tokens[0]].item()]

                # Don't include the prompt.
                logits = logits[:, bias_type_to_position[self._bias_type] :, :]

                output = torch.softmax(logits, dim=-1)

            else:
                joint_sentence_probability = [
                    initial_token_probabilities[0, 0, tokens[0]].item()
                ]

                output = torch.softmax(self._model(tokens_tensor)[0], dim=-1)

        if self._is_self_debias:
            for idx in range(1, len(tokens)):
                joint_sentence_probability.append(
                    output[1, idx - 1, tokens[idx]].item()
                )

        else:
            for idx in range(1, len(tokens)):
                joint_sentence_probability.append(
                    output[0, idx - 1, tokens[idx]].item()
                )

        # Ensure that we have a probability on every token.
        assert len(tokens) == len(joint_sentence_probability)

        score = np.sum([np.log2(i) for i in joint_sentence_probability])
        score /= len(joint_sentence_probability)
        score = np.power(2, score)

        return score

    def _average_log_probability(self, token_ids, spans, device):
        probs = []
        for position in spans:
            # Mask the position.
            masked_token_ids = token_ids.clone().to(device)
            masked_token_ids[:, position] = self._tokenizer.mask_token_id

            with torch.no_grad():
                if self._is_self_debias:
                    # Get the logits for the masked token using self-debiasing.
                    debiasing_prefixes = [DEBIASING_PREFIXES[self._bias_type]]
                    hidden_states = self._model.get_token_logits_self_debiasing(
                        masked_token_ids,
                        debiasing_prefixes=debiasing_prefixes,
                        decay_constant=50,
                        epsilon=0.01,
                    )
                else:
                    hidden_states = self._model(masked_token_ids)["logits"]
                    hidden_states = hidden_states.squeeze(0)
                    hidden_states = hidden_states[position]

            target_id = token_ids[0][position]
            log_probs = F.log_softmax(hidden_states, dim=0)[target_id]
            probs.append(log_probs.item())

        score = np.mean(probs)

        return score


    def _read_data(self, input_file, percentage):
        """Load data into pandas DataFrame format."""

        df_data = pd.DataFrame(columns=["sent1", "sent2", "direction", "bias_type"])

        if self._bias_type is not None:
            print(f"Evaluating {self._bias_type} examples.")

        with open(input_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                direction, gold_bias = "_", "_"
                direction = row["stereo_antistereo"]
                bias_type = row["bias_type"]

                if self._bias_type is not None and bias_type != self._bias_type:
                    continue

                sent1, sent2 = "", ""
                if direction == "stereo":
                    sent1 = row["sent_more"]
                    sent2 = row["sent_less"]
                else:
                    sent1 = row["sent_less"]
                    sent2 = row["sent_more"]

                df_item = {
                    "sent1": sent1,
                    "sent2": sent2,
                    "direction": direction,
                    "bias_type": bias_type,
                }
                
                df_data = pd.concat([df_data, pd.DataFrame([df_item])], ignore_index=True)

            # if certain percentage of data
            df_data = df_data.sample(frac=(percentage/100), random_state=self._seed) 

        return df_data


def _get_span(seq1, seq2, operation):
    """This function extract spans that are shared between two sequences."""
    seq1 = [str(x) for x in seq1.tolist()]
    seq2 = [str(x) for x in seq2.tolist()]

    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    template1, template2 = [], []
    for op in matcher.get_opcodes():
        # each op is a list of tuple:
        # (operation, pro_idx_start, pro_idx_end, anti_idx_start, anti_idx_end)
        # possible operation: replace, insert, equal
        # https://docs.python.org/3/library/difflib.html
        if (operation == "equal" and op[0] == "equal") or (
            operation == "diff" and op[0] != "equal"
        ):
            template1 += [x for x in range(op[1], op[2], 1)]
            template2 += [x for x in range(op[3], op[4], 1)]

    return template1, template2
