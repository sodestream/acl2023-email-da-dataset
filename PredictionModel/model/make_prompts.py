import json
import sys

prompt_template = """
Your task is to find which types of dialog acts occur in an email segment. 
There is a two level hierarchy of dialog acts described below (dialog act types are given in quotes).

1. "InformationProviding" -- Any type of providing information.
1.1. "Agreement" --  Agreeing with opinion or accepting a task.
1.2. "Answer" --  Answering a question.
1.3. "ContextSetting" -- Providing context before other dialog acts.
1.4. "Disagreement" -- Disagreeing with opinion on rejecting a task.
1.5. "Extension" Natural continuation of the previous segment.
1.6. "NeutralResponse" -- Response without clear agreement or disagreement.
1.7. "ProposeAction" -- Propose an actionable activity.
1.8. "StateDecision" -- Explicitly express a decision.
2. "InformationSeeking" -- Any type of seeking information 
2.1. "ClarificationElicitation" Expresses need for further elaboration.
2.2. "Question" Any type of question.
3. "Social" -- Social acts (thanking, apologizing etc.) 
3.1. "Thanking" -- Conveying thanks. Thanks for the comment.

The email segment where you should find dialog acts is given below:

---------------------------------------------------------

[TEXT]

---------------------------------------------------------

Please assign to the above email segment dialog act types given in quotes in the dialog act hierarchy from above. 
You may assign multiple dialog act types if appropriate. 
Where possible you should prefer the more specific dialog act types (second level of the hierarchy) over their more general variants.

Please output your answer as a single line containing a comma-separated list of assigned dialog act types.
"""

def make_prompt(text):
    return prompt_template.replace("[TEXT]", text)

gold_file = sys.argv[1]
prompts_file = sys.argv[2]

gold_json = json.load(open(gold_file, "r"))

json_output = []
for example in gold_json:
    example_text = example["current_segment"]["content"]
    example_id = example["current_segment"]["sid"]
    json_output.append({
          "sid": example_id,
          "prompt": make_prompt(example_text)
        })

json.dump(json_output, open(prompts_file, "w"))




