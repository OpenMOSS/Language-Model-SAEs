"""Prompt builders for generating feature explanations.

This module contains functions for generating prompts used to explain SAE features,
including both neuronpedia-style and OpenAI-style explanation prompts.
"""

from typing import Any

from lm_saes.analysis.autointerp.autointerp_base import AutoInterpConfig
from lm_saes.analysis.samples import TokenizedSample, process_token

NEURONPEDIA_SYSTEM_PROMPT_VANILLA = """You are explaining the behavior of a neuron in a neural network. Your final response should be a very concise explanation (1-6 words) that captures what the neuron detects or predicts by finding patterns in lists.\n\n
To determine the explanation, you are given four lists:\n\n
- MAX_ACTIVATING_TOKENS, which are the top activating tokens in the top activating texts. Each max activating token is shown with the previous 3 tokens in parentheses for context, e.g., "(Who am) I".\n
- TOKENS_AFTER_MAX_ACTIVATING_TOKEN, which are the tokens immediately after the max activating token.\n
- TOP_POSITIVE_LOGITS, which are the most likely words or tokens associated with this neuron.\n
- TOP_ACTIVATING_TEXTS, which are top activating texts.\n\n
You should look for a pattern by trying the following methods in order. You may go through each method even if you find a pattern with some method. You may sometimes need to combine different methods to give a better explanation.\n
Method 1: Look at MAX_ACTIVATING_TOKENS. If they share something specific in common, or are all the same token or a variation of the same token (like different cases or conjugations), respond with that token.
    - Note that MAX_ACTIVATING_TOKENS are preceded by 3 tokens in parentheses as a short context. For example, "(Who am) I" is a max activating token on the word "I" with the context "(Who am)".
    - These preceding tokens can be informative. If all MAX_ACTIVATING_TOKENS have the same or similar preceding tokens, respond with that preceding tokens, e.g. previous is X.
Method 2: Look at TOKENS_AFTER_MAX_ACTIVATING_TOKEN. Try to find a specific pattern or similarity in all the tokens. A common pattern is that they all start with the same letter. If you find a pattern (like \'s word\', \'the ending -ing\', \'number 8\'), respond with \'say [the pattern]\'. You can ignore uppercase/lowercase differences for this.\n
Method 3: Look at TOP_POSITIVE_LOGITS for similarities and describe it very briefly (1-3 words). These tokens are the most likely to be predicted with this neuron.\n
Method 4: Look at TOP_ACTIVATING_TEXTS and make a best guess by describing the broad theme or context, ignoring the max activating tokens.\n\n
Method 5: Look at TOP_NEGATIVE_LOGITS for similarities and describe it very briefly (1-3 words). These tokens are the most suppressed by this neuron. Use this method sparingly. Especially when the neuron is suppressing some rare tokens (e.g. very long tokens starting with newlines and from non-English-or-Chinese alphabets).\n
Rules:\n
- You can think carefully in your internal thinking process, but keep your returned explanation extremely concise (1-6 words, mostly 1-3 words).\n
- Do not add unnecessary phrases like "words related to", "concepts related to", or "variations of the word".\n
- Do not mention "tokens" or "patterns" in your explanation.\n
- The explanation should be specific. For example, "unique words" is not a specific enough pattern, nor is "foreign words".\n
- Remember to use the \'say [the pattern]\' when using Method 2 & 3 above (pattern found in TOKENS_AFTER_MAX_ACTIVATING_TOKEN and TOP_POSITIVE_LOGITS respectively).\n
- Remember to use the \'do not say [the pattern]\' when using Method 5 above (pattern found in TOP_NEGATIVE_LOGITS).\n
- If you absolutely cannot make any guesses, respond with "N/A".\n\n
Think carefully by going through each method number until you find one that helps you find an explanation for what this neuron is detecting or predicting. If a method does not help you find an explanation, briefly explain why it does not, then go on to the next method. Finally, end your thinking process with the method number you used, the reason for your explanation, and return the explanation in a brief manner.\n

Examples:
{
<TOKENS_AFTER_MAX_ACTIVATING_TOKEN>\n\nwas\nwatching\n\n</TOKENS_AFTER_MAX_ACTIVATING_TOKEN>\n\n\n<MAX_ACTIVATING_TOKENS>\n(Who am) I\n(I really) enjoy\n\n</MAX_ACTIVATING_TOKENS>\n\n\n<TOP_POSITIVE_LOGITS>\n\nwalking\nWA\nwaiting\nwas\nwe\nWHAM\nwish\nwin\nwake\nwhisper\n\n</TOP_POSITIVE_LOGITS>\n\n\n<TOP_NEGATIVE_LOGITS>\ndoes\napple\n\\n\nused\nsay\nvitamins\nneus\nautumn\nsun\nanation</TOP_NEGATIVE_LOGITS>\n\n\n<TOP_ACTIVATING_TEXTS>\n\nShe was taking a nap when her phone started ringing.\nI enjoy watching movies with my family.\n\n</TOP_ACTIVATING_TEXTS>\n\n\n<THINKING>Explanation of neuron behavior: \n
Method 1 fails: MAX_ACTIVATING_TOKENS (I, enjoy) are not similar tokens.\nMethod 2 succeeds: All TOKENS_AFTER_MAX_ACTIVATING_TOKEN have a pattern in common: they all start with "w".\nMethod 3 confirms: TOP_POSITIVE_LOGITS also show many words starting with "w" (walking, waiting, was, we, wish, win, wake, whisper), reinforcing the pattern found in Method 2.\nMethod 4: TOP_ACTIVATING_TEXTS don't provide additional clarity beyond what Methods 2 and 3 revealed.\nCombining Methods 2 and 3: The neuron detects tokens that start with "w" and predicts words starting with "w".\nExplanation: say "w" words</THINKING>\n\nsay "w" words
}

{
<TOKENS_AFTER_MAX_ACTIVATING_TOKEN>\n\nwarm\nthe\n\n</TOKENS_AFTER_MAX_ACTIVATING_TOKEN>\n\n\n<MAX_ACTIVATING_TOKENS>\n\n(including you) and\n(matters .) And\n\n</MAX_ACTIVATING_TOKENS>\n\n\n<TOP_POSITIVE_LOGITS>\n\nelephant\nguitar\nmountain\nbicycle\nocean\ntelescope\ncandle\numbrella\ntornado\nbutterfly\n\n</TOP_POSITIVE_LOGITS>\n\n\n<TOP_NEGATIVE_LOGITS>\ndoes\napple\n\\n\nused\nsay\nvitamins\nneus\nautumn\nsun\nanation</TOP_NEGATIVE_LOGITS>\n\n\n<TOP_ACTIVATING_TEXTS>\n\nIt was a beautiful day outside with clear skies and warm sunshine.\nAnd the garden has roses and tulips and daisies and sunflowers blooming together.\n\n</TOP_ACTIVATING_TEXTS>\n\n\n<THINKING>Explanation of neuron behavior: \n
Method 1 succeeds: All MAX_ACTIVATING_TOKENS are the word "and".\nMethod 2: TOKENS_AFTER_MAX_ACTIVATING_TOKEN (warm, the) don't show a clear pattern related to "and".\nMethod 3: TOP_POSITIVE_LOGITS show diverse unrelated words (elephant, guitar, mountain, etc.), not reinforcing the "and" pattern.\nMethod 4: TOP_ACTIVATING_TEXTS show sentences with "and" but don't add information beyond Method 1.\nMethod 1 provides the clearest explanation: the neuron activates on the token "and".\nExplanation: and</THINKING>\n\nand
}

{
<TOKENS_AFTER_MAX_ACTIVATING_TOKEN>\n\nare\n,\n\n</TOKENS_AFTER_MAX_ACTIVATING_TOKEN>\n\n\n<MAX_ACTIVATING_TOKENS>\n\n(from the) banana\n(from the) blueberries\n\n</MAX_ACTIVATING_TOKENS>\n\n\n<TOP_POSITIVE_LOGITS>\n\napple\norange\npineapple\nwatermelon\nkiwi\npeach\npear\ngrape\ncherry\nplum\n\n</TOP_POSITIVE_LOGITS>\n\n\n<TOP_NEGATIVE_LOGITS>\ndoes\napple\n\\n\nused\nsay\nvitamins\nneus\nautumn\nsun\nanation</TOP_NEGATIVE_LOGITS>\n\n\n<TOP_ACTIVATING_TEXTS>\n\nThe apple and banana are delicious foods that provide essential vitamins and nutrients.\nI enjoy eating fresh strawberries, blueberries, and mangoes during the summer months.\n\n</TOP_ACTIVATING_TEXTS>\n\n\n<THINKING>Explanation of neuron behavior: \n
Method 1 succeeds: All MAX_ACTIVATING_TOKENS (banana, blueberries) are fruits.\nMethod 2: TOKENS_AFTER_MAX_ACTIVATING_TOKEN (are, ,) don't show a clear pattern.\nMethod 3 confirms: TOP_POSITIVE_LOGITS show many fruits (apple, orange, pineapple, watermelon, kiwi, peach, pear, grape, cherry, plum), strongly reinforcing the pattern found in Method 1.\nMethod 4: TOP_ACTIVATING_TEXTS mention fruits but don't add information beyond Methods 1 and 3.\nCombining Methods 1 and 3: The neuron activates on fruit tokens and predicts fruit-related words.\nExplanation: fruits\n</THINKING>\n\nfruits
}

{
<TOKENS_AFTER_MAX_ACTIVATING_TOKEN>\n\nwas\nplaces\n\n</TOKENS_AFTER_MAX_ACTIVATING_TOKEN>\n\n\n<MAX_ACTIVATING_TOKENS>\n\n(during the) war\n(in some) places\n\n</MAX_ACTIVATING_TOKENS>\n\n\n<TOP_POSITIVE_LOGITS>\n\n4\nfour\nfourth\n4th\nIV\nFour\nFOUR\n~4\n4.0\nquartet\n\n</TOP_POSITIVE_LOGITS>\n\n\n<TOP_NEGATIVE_LOGITS>\ndoes\napple\n\\n\nused\nsay\nvitamins\nneus\nautumn\nsun\nanation</TOP_NEGATIVE_LOGITS>\n\n\n<TOP_ACTIVATING_TEXTS>\n\nthe civil war was a major topic in history class .\n seasons of the year are winter , spring , summer , and fall or autumn in some places .\n\n</TOP_ACTIVATING_TEXTS>\n\n\n<THINKING>Explanation of neuron behavior: \n
Method 1 fails: MAX_ACTIVATING_TOKENS (war, places) are not all the same token and don't share a clear pattern.\nMethod 2 fails: TOKENS_AFTER_MAX_ACTIVATING_TOKEN (was, places) are not all similar tokens and don't have a text pattern in common.\nMethod 3 succeeds: All TOP_POSITIVE_LOGITS are the number 4 (4, four, fourth, 4th, IV, Four, FOUR, ~4, 4.0, quartet).\nMethod 4: TOP_ACTIVATING_TEXTS mention "war" and "places" but don't clearly relate to the number 4 pattern found in Method 3.\nMethod 5: TOP_NEGATIVE_LOGITS don't show a clear pattern that would help explain the feature.\nMethod 3 provides the clearest explanation: the neuron predicts the number 4.\nExplanation: 4</THINKING>\n\n4
}
"""

NEURONPEDIA_SYSTEM_PROMPT_Z_PATTERN = """You are explaining the behavior of a neuron in a neural network. Your final response should be a very concise explanation (1-6 words) that captures what the neuron detects or predicts by finding patterns in lists.\n\n
To determine the explanation, you are given four lists:\n\n
- MAX_ACTIVATING_TOKENS, which are the top activating tokens in the top activating texts. Each max activating token is shown with the previous 3 tokens in parentheses for context, e.g., "(Who am) I".\n
- TOKENS_AFTER_MAX_ACTIVATING_TOKEN, which are the tokens immediately after the max activating token.\n
- TOP_POSITIVE_LOGITS, which are the most likely words or tokens associated with this neuron.\n
- TOP_ACTIVATING_TEXTS, which are top activating texts.\n\n
You should look for a pattern by trying the following methods in order. You may go through each method even if you find a pattern with some method. You may sometimes need to combine different methods to give a better explanation.\n
Method 1: Look at MAX_ACTIVATING_TOKENS.
    - These neurons are likely to be activated by specific patterns in the text, such as the presence of certain words or phrases. This is much like attention heads attending to a certain concept in the text.
    - The activating pattern is showed in the following format: [(previous tokens) token1] => [(previous tokens) token2]. For example, "[(from Dr.) Sam] => (is from) Dr." is a typical induction head pattern, which moves the attention from "Sam" to "Dr." in the next token.
    - Source tokens and target tokens are preceded by 3 tokens in parentheses as a short context. For example, "(Who am) I" is a max activating token on the word "I" with the context "(Who am)".
    - These preceding tokens can be informative. If all MAX_ACTIVATING_TOKENS have the same or similar preceding tokens, respond with that preceding tokens, e.g. previous is X.
    - If source and target tokens are the same token, this typically means that the neuron is attending to its own token. In this case source token is often not informative. Try to find a pattern in the target token or try other methods.
    - If this method succeeds, try to respond in the format of [source token] => [target token]. For instance, "[position] => [name]".
Method 2: Look at TOKENS_AFTER_MAX_ACTIVATING_TOKEN. Try to find a specific pattern or similarity in all the tokens. A common pattern is that they all start with the same letter. If you find a pattern (like \'s word\', \'the ending -ing\', \'number 8\'), respond with \'say [the pattern]\'. You can ignore uppercase/lowercase differences for this.\n
Method 3: Look at TOP_POSITIVE_LOGITS for similarities and describe it very briefly (1-3 words). These tokens are the most likely to be predicted with this neuron.\n
Method 4: Look at TOP_ACTIVATING_TEXTS and make a best guess by describing the broad theme or context, ignoring the max activating tokens.\n\n
Method 5: Look at TOP_NEGATIVE_LOGITS for similarities and describe it very briefly (1-3 words). These tokens are the most suppressed by this neuron. Use this method sparingly. Especially when the neuron is suppressing some rare tokens (e.g. very long tokens starting with newlines and from non-English-or-Chinese alphabets).\n
Rules:\n
- You can think carefully in your internal thinking process, but keep your returned explanation extremely concise (1-6 words, mostly 1-3 words).\n
- Do not add unnecessary phrases like "words related to", "concepts related to", or "variations of the word".\n
- Do not mention "tokens" or "patterns" in your explanation.\n
- The explanation should be specific. For example, "unique words" is not a specific enough pattern, nor is "foreign words".\n
- Remember to use the \'say [the pattern]\' when using Method 2 & 3 above (pattern found in TOKENS_AFTER_MAX_ACTIVATING_TOKEN and TOP_POSITIVE_LOGITS respectively).\n
- Remember to use the \'do not say [the pattern]\' when using Method 5 above (pattern found in TOP_NEGATIVE_LOGITS).\n
- If you absolutely cannot make any guesses, respond with "N/A".\n\n
Think carefully by going through each method number until you find one that helps you find an explanation for what this neuron is detecting or predicting. If a method does not help you find an explanation, briefly explain why it does not, then go on to the next method. Finally, end your thinking process with the method number you used, the reason for your explanation, and return the explanation in a brief manner.\n

Examples:
{
<TOKENS_AFTER_MAX_ACTIVATING_TOKEN>\n\nwas\nwatching\n\n</TOKENS_AFTER_MAX_ACTIVATING_TOKEN>\n\n\n<MAX_ACTIVATING_TOKENS>\n[(Who am) I] => (Who am) I\n[(I really) enjoy] => (I really) enjoy\n\n</MAX_ACTIVATING_TOKENS>\n\n\n<TOP_POSITIVE_LOGITS>\n\nwalking\nWA\nwaiting\nwas\nwe\nWHAM\nwish\nwin\nwake\nwhisper\n\n</TOP_POSITIVE_LOGITS>\n\n\n<TOP_NEGATIVE_LOGITS>\ndoes\napple\n\\n\nused\nsay\nvitamins\nneus\nautumn\nsun\nanation</TOP_NEGATIVE_LOGITS>\n\n\n<TOP_ACTIVATING_TEXTS>\n\nShe was taking a nap when her phone started ringing.\nI enjoy watching movies with my family.\n\n</TOP_ACTIVATING_TEXTS>\n\n\n<THINKING>Explanation of neuron behavior: \n
Method 1 fails: MAX_ACTIVATING_TOKENS show self-attention patterns [(Who am) I] => (Who am) I and [(I really) enjoy] => (I really) enjoy, but the source tokens (I, enjoy) are not similar.\nMethod 2 succeeds: All TOKENS_AFTER_MAX_ACTIVATING_TOKEN have a pattern in common: they all start with "w".\nMethod 3 confirms: TOP_POSITIVE_LOGITS also show many words starting with "w" (walking, waiting, was, we, wish, win, wake, whisper), reinforcing the pattern found in Method 2.\nMethod 4: TOP_ACTIVATING_TEXTS don't provide additional clarity beyond what Methods 2 and 3 revealed.\nCombining Methods 2 and 3: The neuron detects tokens that start with "w" and predicts words starting with "w".\nExplanation: say "w" words</THINKING>\n\nsay "w" words
}

{
<TOKENS_AFTER_MAX_ACTIVATING_TOKEN>\n\nManning\nChris\n\n</TOKENS_AFTER_MAX_ACTIVATING_TOKEN>\n\n\n<MAX_ACTIVATING_TOKENS>\n\n[(from Dr.) Sam] => (is from) Dr.]\n[(is Prof.) Chris] => (he is) Prof.\n\n</MAX_ACTIVATING_TOKENS>\n\n\n<TOP_POSITIVE_LOGITS>\n\nelephant\nguitar\nmountain\nbicycle\nocean\ntelescope\ncandle\numbrella\ntornado\nbutterfly\n\n</TOP_POSITIVE_LOGITS>\n\n\n<TOP_NEGATIVE_LOGITS>\ndoes\napple\n\\n\nused\nsay\nvitamins\nneus\nautumn\nsun\nanation</TOP_NEGATIVE_LOGITS>\n\n\n<TOP_ACTIVATING_TEXTS>\n\nIt was a beautiful day outside with clear skies and warm sunshine.\nAnd the garden has roses and tulips and daisies and sunflowers blooming together.\n\n</TOP_ACTIVATING_TEXTS>\n\n\n<THINKING>Explanation of neuron behavior: \n
Method 1 succeeds: Looking at the MAX_ACTIVATING_TOKENS, we can see that this neuron is attending to the position and name of the person. The pattern is [(from Dr.) Sam] => (is from) Dr. and [(is Prof.) Chris] => (he is) Prof., showing attention from name to position.\nMethod 2: TOKENS_AFTER_MAX_ACTIVATING_TOKEN (Manning, Chris) are names, which aligns with the source tokens in Method 1.\nMethod 3: TOP_POSITIVE_LOGITS show diverse unrelated words, not reinforcing the position-name pattern.\nMethod 4: TOP_ACTIVATING_TEXTS don't provide additional clarity beyond Method 1.\nMethod 1 provides the clearest explanation: the neuron attends from name to position.\nExplanation: [name] => [position]</THINKING>\n\n[name] => [position]
}

{
<TOKENS_AFTER_MAX_ACTIVATING_TOKEN>\n\nare\n,\n\n</TOKENS_AFTER_MAX_ACTIVATING_TOKEN>\n\n\n<MAX_ACTIVATING_TOKENS>\n\n[(from the) banana] => (from the) banana\n[(from the) blueberries] => (from the) blueberries\n\n</MAX_ACTIVATING_TOKENS>\n\n\n<TOP_POSITIVE_LOGITS>\n\napple\norange\npineapple\nwatermelon\nkiwi\npeach\npear\ngrape\ncherry\nplum\n\n</TOP_POSITIVE_LOGITS>\n\n\n<TOP_NEGATIVE_LOGITS>\ndoes\napple\n\\n\nused\nsay\nvitamins\nneus\nautumn\nsun\nanation</TOP_NEGATIVE_LOGITS>\n\n\n<TOP_ACTIVATING_TEXTS>\n\nThe apple and banana are delicious foods that provide essential vitamins and nutrients.\nI enjoy eating fresh strawberries, blueberries, and mangoes during the summer months.\n\n</TOP_ACTIVATING_TEXTS>\n\n\n<THINKING>Explanation of neuron behavior: \n
Method 1 succeeds: All MAX_ACTIVATING_TOKENS show self-attention on fruit tokens [(from the) banana] => (from the) banana and [(from the) blueberries] => (from the) blueberries. The tokens (banana, blueberries) are fruits.\nMethod 2: TOKENS_AFTER_MAX_ACTIVATING_TOKEN (are, ,) don't show a clear pattern.\nMethod 3 confirms: TOP_POSITIVE_LOGITS show many fruits (apple, orange, pineapple, watermelon, kiwi, peach, pear, grape, cherry, plum), strongly reinforcing the pattern found in Method 1.\nMethod 4: TOP_ACTIVATING_TEXTS mention fruits but don't add information beyond Methods 1 and 3.\nCombining Methods 1 and 3: The neuron activates on fruit tokens and predicts fruit-related words.\nExplanation: fruits\n</THINKING>\n\nfruits
}

{
<TOKENS_AFTER_MAX_ACTIVATING_TOKEN>\n\nwas\nplaces\n\n</TOKENS_AFTER_MAX_ACTIVATING_TOKEN>\n\n\n<MAX_ACTIVATING_TOKENS>\n\n[(during the) war] => (during the) war\n[(in some) places] => (in some) places\n\n</MAX_ACTIVATING_TOKENS>\n\n\n<TOP_POSITIVE_LOGITS>\n\n4\nfour\nfourth\n4th\nIV\nFour\nFOUR\n~4\n4.0\nquartet\n\n</TOP_POSITIVE_LOGITS>\n\n\n<TOP_NEGATIVE_LOGITS>\ndoes\napple\n\\n\nused\nsay\nvitamins\nneus\nautumn\nsun\nanation</TOP_NEGATIVE_LOGITS>\n\n\n<TOP_ACTIVATING_TEXTS>\n\nthe civil war was a major topic in history class .\n seasons of the year are winter , spring , summer , and fall or autumn in some places .\n\n</TOP_ACTIVATING_TEXTS>\n\n\n<THINKING>Explanation of neuron behavior: \n
Method 1 fails: MAX_ACTIVATING_TOKENS show self-attention patterns [(during the) war] => (during the) war and [(in some) places] => (in some) places, but the tokens (war, places) are not all the same token and don't share a clear pattern.\nMethod 2 fails: TOKENS_AFTER_MAX_ACTIVATING_TOKEN (was, places) are not all similar tokens and don't have a text pattern in common.\nMethod 3 succeeds: All TOP_POSITIVE_LOGITS are the number 4 (4, four, fourth, 4th, IV, Four, FOUR, ~4, 4.0, quartet).\nMethod 4: TOP_ACTIVATING_TEXTS mention "war" and "places" but don't clearly relate to the number 4 pattern found in Method 3.\nMethod 5: TOP_NEGATIVE_LOGITS don't show a clear pattern that would help explain the feature.\nMethod 3 provides the clearest explanation: the neuron predicts the number 4.\nExplanation: 4</THINKING>\n\n4
}
"""


def generate_explanation_prompt_neuronpedia(
    cfg: AutoInterpConfig,
    activating_examples: list[TokenizedSample],
    top_logits: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[str, str]:
    """Generate a prompt for explanation generation with neuronpedia.

    Args:
        cfg: Auto-interpretation configuration
        activating_examples: List of activating examples
        top_logits: Optional top logits dictionary with 'top_positive' and 'top_negative' keys

    Returns:
        Tuple of (system_prompt, user_prompt) strings
    """
    system_prompt = (
        NEURONPEDIA_SYSTEM_PROMPT_Z_PATTERN
        if activating_examples[0].has_z_pattern_data()
        else NEURONPEDIA_SYSTEM_PROMPT_VANILLA
    )
    examples_to_show = activating_examples[: cfg.n_activating_examples]
    next_activating_tokens = ""
    max_activating_tokens = ""
    plain_activating_tokens = ""
    logit_activating_tokens = ""
    logit_suppressing_tokens = ""

    for i, example in enumerate(examples_to_show, 1):
        next_activating_tokens = next_activating_tokens + example.display_next(cfg.activation_threshold)
        max_activating_tokens = max_activating_tokens + example.display_max(cfg.activation_threshold)
        plain_activating_tokens = plain_activating_tokens + process_token(example.display_plain()) + "\n"

    if top_logits is not None:
        for text in top_logits["top_positive"]:
            logit_activating_tokens = logit_activating_tokens + process_token(text["token"]) + "\n"
        for text in top_logits["top_negative"]:
            logit_suppressing_tokens = logit_suppressing_tokens + process_token(text["token"]) + "\n"
    else:
        logit_activating_tokens = next_activating_tokens
        logit_suppressing_tokens = "none"

    user_prompt: str = f"""
<TOKENS_AFTER_MAX_ACTIVATING_TOKEN>\n\n{next_activating_tokens}\n</TOKENS_AFTER_MAX_ACTIVATING_TOKEN>\n\n\n<MAX_ACTIVATING_TOKENS>\n\n{max_activating_tokens}\n</MAX_ACTIVATING_TOKENS>\n\n\n<TOP_POSITIVE_LOGITS>\n\n{logit_activating_tokens}\n</TOP_POSITIVE_LOGITS>\n\n\n<TOP_NEGATIVE_LOGITS>\n\n{logit_suppressing_tokens}\n</TOP_NEGATIVE_LOGITS>\n\n\n<TOP_ACTIVATING_TEXTS>\n\n{plain_activating_tokens}\n<\\TOP_ACTIVATING_TEXTS>\n\n\nExplanation of neuron behavior: \n
"""
    # print('system_prompt', system_prompt)
    # print('user_prompt', user_prompt)
    return system_prompt, user_prompt


def generate_explanation_prompt(
    cfg: AutoInterpConfig,
    activating_examples: list[TokenizedSample],
) -> tuple[str, str]:
    """Generate a prompt for explanation generation.

    Args:
        cfg: Auto-interpretation configuration
        activating_examples: List of activating examples

    Returns:
        Tuple of (system_prompt, user_prompt) strings
    """
    cot_prompt = ""
    if cfg.include_cot:
        cot_prompt += "\n\nTo explain this feature, please follow these steps:\n"
        cot_prompt += "Step 1: List a couple activating and contextual tokens you find interesting. "
        cot_prompt += "Search for patterns in these tokens, if there are any. Don't list more than 5 tokens.\n"
        cot_prompt += "Step 2: Write down general shared features of the text examples.\n"
        cot_prompt += "Step 3: Write a concise explanation of what this feature detects.\n"

    examples_prompt = """Some examples:

{
    "steps": ["Activating token: <<knows>>. Contextual tokens: Who, ?. Pattern: <<knows>> is consistently activated, often found in sentences starting with interrogative words like 'Who' and ending with a question mark.", "Shared features include consistent activation on the word 'knows'. The surrounding text always forms a question. The questions do not seem to expect a literal answer, suggesting they are rhetorical.", "This feature activates on the word knows in rhetorical questions"],
    "final_explanation": "The feature activates on the word 'knows' in rhetorical questions.",
    "activation_consistency": 5,
    "complexity": 4
}

{
    "steps": ["Activating tokens: <<Entwickler>>, <<Enterprise>>, <<Entertainment>>, <<Entity>>, <<Entrance>>. Pattern: All activating instances are on words that begin with the specific substring 'Ent'. The activation is on the 'Ent' portion itself.", "The shared feature across all examples is the presence of words starting with the capitalized substring 'Ent'. The feature appears to be case-sensitive and position-specific (start of the word). No other contextual or semantic patterns are observed."],
    "final_explanation": "The feature activates on the substring 'Ent' at the start of words",
    "activation_consistency": 5,
    "complexity": 1
}

{
    "steps": ["Activating tokens: <<budget deficit>>, <<interest rates>>, <<fiscal stimulus>>, <<trade policy>>, <<unemployment benefits>>. Pattern: Activations highlight phrases and concepts central to economic discussions and government actions.","The examples consistently involve discussions of economic indicators, government spending, financial regulation, or international trade agreements. While most activations clearly relate to economic policies enacted or debated by governmental bodies, some activations might be on broader economic news or expert commentary where the direct link to a specific government policy is less explicit, or on related but not identical topics like corporate financial health in response to policy."],
    "final_explanation": "The feature activates on text about government economic policy",
    "activation_consistency": 3,
    "complexity": 5
}

"""
    system_prompt: str = f"""We're studying features in a neural network. Each feature activates on some particular word/words/substring/concept in a short document. The activating words in each document are indicated with << ... >>. We will give you a list of documents on which the feature activates, in order from most strongly activating to least strongly activating.

Your task is to:

First, Summarize the Activation: Look at the parts of the document the feature activates for and summarize in a single sentence what the feature is activating on. Try not to be overly specific in your explanation. Note that some features will activate only on specific words or substrings, but others will activate on most/all words in a sentence provided that sentence contains some particular concept. Your explanation should cover most or all activating words (for example, don't give an explanation which is specific to a single word if all words in a sentence cause the feature to activate). Pay attention to things like the capitalization and punctuation of the activating words or concepts, if that seems relevant. Keep the explanation as short and simple as possible, limited to 20 words or less. Omit punctuation and formatting. You should avoid giving long lists of words.{cot_prompt}

Second, Assess Activation Consistency: Based on your summary and the provided examples, evaluate the consistency of the feature's activation. Return your assessment as a single integer from the following scale:

5: Clear pattern with no deviating examples
4: Clear pattern with one or two deviating examples
3: Clear overall pattern but quite a few examples not fitting that pattern
2: Broad consistent theme but lacking structure
1: No discernible pattern

Third, Assess Feature Complexity: Based on your summary and the nature of the activation, evaluate the complexity of the feature. Return your assessment as a single integer from the following scale:

5: Rich feature firing on diverse contexts with an interesting unifying theme, e.g., "feelings of togetherness"
4: Feature relating to high-level semantic structure, e.g., "return statements in code"
3: Moderate complexity, such as a phrase, category, or tracking sentence structure, e.g., "website URLs"
2: Single word or token feature but including multiple languages or spelling, e.g., "mentions of dog"
1: Single token feature, e.g., "the token '('"

Your output should be a JSON object that has the following fields: `steps`, `final_explanation`, `activation_consistency`, `complexity`. `steps` should be an array of strings with a length not exceeding 3, each representing a step in the chain-of-thought process. `final_explanation` should be a string in the form of 'This feature activates on... '. `activation_consistency` should be an integer between 1 and 5, representing the consistency of the feature. `complexity` should be an integer between 1 and 5, representing the complexity of the feature.

{examples_prompt}
"""

    user_prompt = "The activating documents are given below:\n\n"
    # Select a subset of examples to show
    examples_to_show = activating_examples[: cfg.n_activating_examples]

    for i, example in enumerate(examples_to_show, 1):
        highlighted = example.display_highlighted(cfg.activation_threshold)
        user_prompt += f"Example {i}: {highlighted}\n\n"

    return system_prompt, user_prompt
