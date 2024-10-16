from typing import List, Dict 
from bias_bench.adaptation.prompt_context import get_stereo_context
import re
import json 

def get_inter_stereo_prompt(instances) -> List[str]:
    model_prompts = list()
    for instance in instances:
        bias_type = instance['bias_type']
        target = instance['target']

        background_info = get_stereo_context(bias_type).replace("\n", " ")

        model_prompts += [f"""Please adapt the StereoSet Bias dataset from its Western-oriented context to a Singaporean context. Objectives: - Adapt wording to align with the Singaporean context, focusing on **race, gender, religion, and profession**. **Adaptation Guidelines**: - target: Modify to a relevant group in Singapore if applicable; this field is optional for adaptation. - bias_type: Keep unchanged. - context: Adjust to reflect situations common in Singapore. - sentences and labels: Rewrite each sentence to align with the Singaporean context, preserving its intended bias category (stereotype, anti-stereotype, unrelated). Ensure each label is correctly assigned, and each bias type is used only once. **Handling Difficult Adaptations**: - If adaptation is challenging, leave the instance unchanged and provide a reason (maximum 50 words) in the format: `"reason: [your reason]"`. If you helped adapt the sentences, state `"reason: NA"`. **Singapore Bias Profile**: - {background_info} **Adaptation Instructions**: Given the following JSON object from the StereoSet dataset, provide the adapted version in the same format, ensuring each sentence has its label correctly assigned and including the `"reason"` field: {{ "target": "{target}", "bias_type": "{bias_type}", "context": "{instance['context']}", "sentences": [ {{"sentence": "{instance['sentences'][0]['sentence']}", "label": "{instance['sentences'][0]['gold_label']}"}}, {{"sentence": "{instance['sentences'][1]['sentence']}", "label": "{instance['sentences'][1]['gold_label']}"}}, {{"sentence": "{instance['sentences'][2]['sentence']}", "label": "{instance['sentences'][2]['gold_label']}"}} ], "reason": "[Your reason or 'NA']"}} Provide the adapted version in the above JSON format after 'Your answer: '. Your answer: JSON""".strip()]


    return model_prompts



def get_intra_stereo_prompt(instances) -> List[str]:
    model_prompts = list()
    for instance in instances:
        bias_type = instance['bias_type']
        target = instance['target']
        model_prompts += [f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        
        You are a helpful AI assistant for adapting StereoSet Bias benchmark dataset from United States (USA) context to Singaporean (SG) context. The StereoSet dataset is designed to measure social biases in Large Language Models (LLMs). In each sample, one sentence expresses a stereotype about the target group, while the other sentence expresses an anti-stereotypical about the target group, and the last sentence expresses an unrelated (meaningless) option. The intrasentence task measures the bias and the language modeling ability at sentence-level. The original task is to choose the appropriate word to fill in the BLANK.  
                           
        Your task is to refine all sentences by adapting them into a Singaporean context. You may adjust the scenarios to make them more relevant locally, but maintain a strong focus on addressing the bias_type being discussed. Ensure the examples and language reflect Singapore's social, cultural, and technological environment.                            
                          
        Here is the {bias_type} profile of Singapore:
        {get_stereo_context(bias_type)}<|eot_id|><|start_header_id|>user<|end_header_id|>
                                                    
        Each StereoSet sample contains:
        target: The group or individual that is the subject of stereotyping.
        bias_type: This column categorizes the type of bias expressed in the sentence pair. It can be one of the 3 categories like race, gender, religion and profession.
        context: The sentence template with a BLANK placeholder to be filled in when generating the three sentences.
        sentence_1: A completed sentence filling in the BLANK placeholder.
        label_1: The label for sentence_1 (stereotype, anti-stereotype, or unrelated)
        sentence_2: Another completed sentence filling in the BLANK placeholder.
        label_2: The label for sentence_2 (stereotype, anti-stereotype, or unrelated)
        sentence_3: A third completed sentence filling in the BLANK placeholder.
        label_3: The label for sentence_3 (stereotype, anti-stereotype, or unrelated)


        Please adapt the following StereoSet sample to the Singaporean context. Try your best to adapt each category accordingly. If you find it difficult to adapt to the context, leave everything as is and provide a reason using the format: 'Reason: [put your reason here]' at the end of your response. If no adaptation issues are found, simply write 'Reason: NA'.
        
        target: {target}
        bias_type: {bias_type}
        context: {instance['context']}
        sentence_1: {instance['sentences'][0]['sentence']}
        label_1: {instance['sentences'][0]['gold_label']}
        sentence_2: {instance['sentences'][1]['sentence']}
        label_2: {instance['sentences'][1]['gold_label']}
        sentence_3: {instance['sentences'][2]['sentence']}
        label_3: {instance['sentences'][2]['gold_label']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""".strip()
    ]
        
    return model_prompts




def get_crows_prompt(instances) -> List[str]:
    model_prompts = list()
    for index, instance in instances.iterrows():
        bias_type = instance['bias_type']
        bias_context_get = bias_type
        if bias_type == "race_color":
            bias_context_get = "race"
        if bias_type == 'nationality':
            bias_context_get = 'race'

        background_info = get_stereo_context(bias_context_get).replace("\n", " ")

        model_prompts += [f"""Please adapt the Crows-Pair Bias dataset from its Western-oriented context to a Singaporean context. **Objectives**: Adapt the wording to align with the Singaporean context, focusing on **age, disability, race-color, gender, nationality, physical-appearance, sexual-orientation, religion, and socioeconomic** factors. **Adaptation Guidelines**: - **bias_type**: Keep unchanged. - **Sent_more**: Rewrite the sentence with more stereotyping or bias (typically regarding a disadvantaged or minority group) to fit the Singaporean context. - **Sent_less**: Rewrite the less biased version of `sent_more`, typically similar except for identifying a more advantaged group (only words identifying the group should differ from `sent_more`). - **Stereo_antistereo**: Adjust to reflect whether `sent_more` expresses a stereotype ("stereo") or opposes it ("antistereo"). **Handling Difficult Adaptations**: If adaptation is challenging, leave the instance unchanged and provide a reason (max 50 words) in the format: "reason: [your reason]". If no issues are found, use "reason: NA". **Singapore Bias Profile**: {background_info} **Adaptation Instructions**: Given the following JSON object from the Crows-Pair dataset, provide the adapted version in the same format, ensuring each sentence has its label correctly assigned and including the "reason" field: {{ "bias_type": "{bias_type}", "sent_more": "{instance['sent_more']}", "sent_more": "{instance['sent_less']}", "stereo_antistereo": "{instance['stereo_antistereo']}", "reason": "[Your reason or 'NA']"}} Provide the adapted version in the above JSON format after 'Your answer: '. Your answer: JSON""".strip()]
    return model_prompts





def extract_sample_from_response(dataset, response, retry) -> Dict:

    if dataset.lower() == "stereoset":
        results = {
            'context': '',
            'target': '',
            'bias_type': '',
            'sentence_1': '',
            'label_1': '',
            'sentence_2': '',
            'label_2': '',
            'sentence_3': '',
            'label_3': '',
            'reason': '',
            }

    if dataset.lower() == "crows":
        results = {
            'bias_type': '',
            'sent_more': '',
            'sent_less': '',
            'stereo_antistereo': '',
            'reason': '',
        }

    try:
        # print("222")
        result_json = re.sub(f'^.*?{re.escape("{")}', '{', response)

        json_match = re.search(r'\{.*\}', response, re.DOTALL)


        json_string = json_match.group(0)
        result_json = json.loads(json_string)  # Convert JSON string to Python dictionary

        # print('333')
        # result_json = json.loads(result_json)
        # print(result_json)

        if dataset.lower() == "stereoset":
            results['context'] = result_json['context']
            results['target'] = result_json['target']
            results['bias_type'] = result_json['bias_type']

            results['sentence_1'] = result_json['sentences'][0]['sentence']
            results['sentence_2'] = result_json['sentences'][1]['sentence']
            results['sentence_3'] = result_json['sentences'][2]['sentence']

            results['label_1'] = result_json['sentences'][0]['label']
            results['label_2'] = result_json['sentences'][1]['label']
            results['label_3'] = result_json['sentences'][2]['label']

            results['reason'] = result_json['reason']

        if dataset.lower() == "crows":
            results['bias_type'] = result_json['bias_type']
            results['sent_more'] = result_json['sent_more']
            results['sent_less'] = result_json['sent_less']
            results['stereo_antistereo'] = result_json['stereo_antistereo']
            results['reason'] = result_json['reason']

    except:
        print(f"retry: {retry}")
        pass

    return results
