# E5 controlled benchmark

This is an author-defined benchmark for evaluating natural-language planning-intent parsing. It is not described as crowdsourced, expert-annotated, or independently human-annotated.

- 20 simple, complete requests.
- 20 composite, complete requests.
- 20 ambiguous or invalid requests.
- One request represents one MHA-PM invocation and therefore permits only one time scenario and one hazard scenario.
- Missing or genuinely ambiguous information is labelled `needs_clarification`.
- Explicit conflicts, invalid facility numbers, unsupported objectives, and instructions for the LLM to directly select final locations are labelled `invalid`.
- Valid requests use only `weighted_p_median`, with integer `p` from 5 through 15.

Benchmark construction is separated from inference. Inference code reads only `request_id`, `request_text`, and `request_type`; gold labels are loaded only by the evaluation stage.
