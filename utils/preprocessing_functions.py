import re
import random
import string
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

# Optional: if you plan to use the random and math-based logic in your earlier functions
import re
import random
import string
from fractions import Fraction

# If you are on Python 3.9+, you can use the built-in tuple/list for type hints.
# Otherwise, use the capitalized versions from typing.
class TextAugmenter:
    """Augments student explanations with realistic variations and performs
    simple word-to-number and number-to-word conversions based on manual maps."""

    def __init__(self, augmentation_prob=0.3):
        # ... (other init parts remain the same)
        self.check=True
        self.augmentation_prob = augmentation_prob
        self.duplicate_chars = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]

# Alternatively, a more concise way to generate the alphabet:
# import string
# self.duplicate_chars = list(string.ascii_lowercase)
        self.synonyms = {
    # --- Existing Synonyms ---
    'since': ['because', 'as', 'given that'], 'because': ['since', 'as', 'cause',''],
    'so': ['therefore', 'thus', 'hence'], 'therefore': ['so', 'thus'],
    'multiply': ['times', 'multiplied by', 'multiply'], 'times': ['multiply', 'multiplied by', 'x', 'X'],
    'divide': ['divided by', 'over'], 'divided by': ['divide', 'over'],
    'add': ['plus', 'added to', 'add', 'sum'], 'plus': ['add', 'added to'], # 'sum' added from D
    'subtract': ['minus', 'take away'], 'minus': ['subtract', 'take away'],

    'equals': ['is', 'is equal to', '='], 'answer': ['result', 'solution'],
    'result': ['answer', 'solution'], 'get': ['got', 'obtain', 'end up with'], 'got': ['get', 'obtained'],

    # --- New Synonyms from D (Merged and Cleaned) ---
    'would': ['should', ""],
    'I think': [""],
    'added': ['summed'],
    'numerator': ['top'],
    'denominator': ['bottom', 'denimnater', 'denominar'],
    'thought': ['think'],
    'numbers': ['values'],
    'number': ['value'],
    'denominators': ['bottoms', 'denimnaters','denominars'],
    'numerators': ['tops'],
    "'i've": ['I have', 'i have'],
    'found': ['find'],
    'to minuses': ['two minuses'],
    'minises': ['minuses'],
    'minise': ['minus'],
    'minus': ['subtract', 'minise'],
    'shaded': ['filled in', 'filled'],
    'information': ['info'],
    'info': ['information'],
    'reciprocal': ['multiplicative inverse'],
    'small': ['tiny'],
    'lcm': ['least commmon multiple', 'lowest common multiple'],
    'lcd': ['least common denominator', 'lowest common multiple'],
    'LCM': ['least commmon multiple', 'lowest common multiple'],
    'LCD': ['least common denominator', 'lowest common multiple'],
    'LCM': ['least commmon multiple', 'lowest common multiple'],


    # --- Consolidating Overlap/Related Terms ---
    # The initial 'add' and 'subtract' entries were already quite robust, but they
    # benefit from merging:
    'subtract': ['minus', 'take away', 'minise'], # Added 'minise' from D
    'amount is reducing':['amount is going down'],
    'amount is increasing':['amount is going up'],
    'bigger': ['larger','greater','Larger','Greater'],
    'smaller':['less','Less','Smaller','lower'],
    'less':['smaller','Smaller','Less','fewer'],
    'biggest':['largest','greatest','Largest','Greatest'],
    'larger':['bigger','Larger','Bigger'],
    'smallest':['Smallest','least','Least'],
    'big':['large','Big','Large'],
    'least':['lowest','Least','Lowest','smallest','Smallest'],
    'convert':['Convert','change','Change'],
    'i think':[''],
    'i believe': [''],
    'I believe':[''],
    'fewer': ['less'],



}
        self.number_words = {
            '0': 'zero', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
            '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen',
            '14': 'fourteen', '15': 'fifteen', '16': 'sixteen', '17': 'seventeen',
            '18': 'eighteen', '19': 'nineteen', '20': 'twenty',
            '30': 'thirty', '40': 'forty', '50': 'fifty', '60': 'sixty',
            '70': 'seventy', '80': 'eighty', '90': 'ninety', '100': 'hundred'
        }

        self.word_to_number_map = {v: k for k, v in self.number_words.items()}
        # NOTE: Fraction words are kept, but primarily for the word-to-number function's sake
        self.fraction_number_words = r'(?:one|two|three|four|five|six|seven|eight|nine|ten|half|quarter)'
        self.fraction_word_map = {
            # Thirds
            'one third': '1/3', 'a third': '1/3', '2 thirds': '2/3', '2 THIRDS': '2/3', 'two thirds': '2/3',
            # Halves
            'one half': '1/2', 'a half': '1/2', '1 half': '1/2', 'ONE HALF': '1/2',
            # Quarters (Fourths)
            'one quarter': '1/4', 'a quarter': '1/4', '1 fourth': '1/4', 'three quarters': '3/4',
            '3 quarters': '3/4', 'three fourths': '3/4', '3 FOURTHS': '3/4',
            # Fifths
            'one fifth': '1/5', 'a fifth': '1/5', '2 fifths': '2/5', 'two fifths': '2/5', '3 fifths': '3/5',
            'three fifths': '3/5', '4 fifths': '4/5', 'four fifths': '4/5'
        }
        # Create a combined regex pattern for faster searching of fractional words
        self.fraction_word_pattern = re.compile('|'.join(re.escape(k) for k in self.fraction_word_map.keys()), re.IGNORECASE)

    # ... (apply_typo, substitute_synonym, number_to_word, word_to_number, remove_punctuation, add_filler_words methods remain the same)
    def substitute_synonym3(self,text):
        if self.check:
            print('this function is called hello')
            self.check=False
        for word, synonyms in self.synonyms.items():
            sub=random.choice(synonyms)

            #word_to_sub=" "+word+" "
            #sub=" "+sub+" "
            #print(word_to_sub, sub)
            text=re.sub(word,sub,text)
        return text


    def substitute_synonym(self, text: str) -> str:
        # 1. Iterate over ALL synonym keys, not just the ones found in the text first.
        # This is often simpler than trying to pre-tokenize the text.

        # We set this to 1.0 to ensure replacement attempts on every eligible word
        REPLACEMENT_PROBABILITY = 0.4

        for word_to_replace, replacements in self.synonyms.items():
            # Skip if the key has no replacements
            if not replacements:
                continue

            # Apply the per-word probability check
            if random.random() < REPLACEMENT_PROBABILITY:

                # 2. Select a replacement
                replacement = random.choice(replacements)

                # 3. Define the SAFE pattern using word boundaries (\b) and ignoring case
                # re.escape is essential to handle keys like 'I have' or 'minuses'.
                pattern_sub = re.compile(r'\b' + re.escape(word_to_replace) + r'\b', re.IGNORECASE)

                # 4. Define the function for preserving case
                # This is done inside the loop to capture the 'replacement' string via closure
                def replace_preserving_case(match):
                    original = match.group(0)
                    if original.isupper():
                      return original
                    # If the first character of the matched word is uppercase, capitalize the replacement
                    if original and original[0].isupper():
                        return replacement.capitalize()
                    return replacement

                # 5. Perform the substitution
                # count=1 ensures we only replace the FIRST occurrence of the word found.
                text = pattern_sub.sub(replace_preserving_case, text, count=1)

        return text
    def apply_typo(self, text: str) -> str:
        words = text.split()
        if not words: return text
        valid_indices = [i for i, w in enumerate(words) if not w.isupper()]

        if not valid_indices:
            return text

        word_idx = random.choice(valid_indices)
        word = words[word_idx]

        valid_chars = [i for i, c in enumerate(word) if c.lower() in self.duplicate_chars]
        if valid_chars:
            char_idx = random.choice(valid_chars)
            words[word_idx] = word[:char_idx+1] + word[char_idx] + word[char_idx+1:]
        return ' '.join(words)

    # Assuming self.synonyms is the large dictionary you provided
    # Assuming the rest of the TextAugmenter class is intact
    def randomly_capitalize_word(self, text: str) -> str:
      """
      Randomly capitalizes a single word that is NOT a feature name (not ALL CAPS).
      """
      # 1. Overall probability check (25% chance to even try)
      if random.random() > 0.75:
          return text

      words = text.split()
      if len(words) < 2:
          return text

      # 2. Identify indices that are NOT the first word AND NOT already ALL CAPS
      # This protects your feature names like TOTAL or RATIO_SIM
      valid_indices = [i for i in range(1, len(words)) if not words[i].isupper()]

      if not valid_indices:
          return text

      # 3. Pick one of the SAFE indices
      word_idx = random.choice(valid_indices)

      # 4. Decide: FULL CAPS (20% chance) or just Capitalized (80% chance)
      if random.random() < 0.2:
          words[word_idx] = words[word_idx].upper()
      else:
          words[word_idx] = words[word_idx].capitalize()

      return ' '.join(words)

    def number_to_word(self, text: str) -> str:
        # (Implementation omitted for brevity)
        if random.random() > 0.4: return text
        pattern = r'(?<![./])\b(\d+)\b(?![./])'
        def replace_number(match):
            num = match.group(1)
            if num in self.number_words: return self.number_words[num]
            return num
        text = re.sub(pattern, replace_number, text, count=1)
        return text

    def word_to_number(self, text: str) -> str:
        # (Implementation omitted for brevity)
        words = text.split()
        if not words: return text
        new_words = []
        replaced = False
        for word in words:
            clean_word = word.lower().strip('.,!?;')
            if clean_word in self.word_to_number_map:
                digit = self.word_to_number_map[clean_word]
                if word != clean_word and not word[-1].isalnum():
                    digit += word[len(clean_word)]
                new_words.append(digit)
                replaced = True
            else:
                new_words.append(word)
        return ' '.join(new_words)

    def remove_punctuation(self, text: str) -> str:
        # Only apply the removal a percentage of the time (e.g., 30%)
        if random.random() > 0.7:
            return text

        # List of punctuation marks to randomly remove *once* from the text
        punctuations = [',', ':', ';']

        # Choose one punctuation to remove for this run
        punc_to_remove = random.choice(punctuations)

        # Find the first occurrence of that punctuation and remove it
        if punc_to_remove in text:
            # We use re.sub for a more robust removal, but simple replace works too
            # for a single instance. Using simple replace for clarity:
            text = text.replace(punc_to_remove, '', 1)

        return text

    def add_filler_words(self, text: str) -> str:
        # (Implementation omitted for brevity)
        if random.random() > 0.9: return text
        fillers = ['like', 'just', 'basically', 'actually', 'really']
        filler = random.choice(fillers)
        words = text.split()
        if len(words) > 3:
            insert_pos = random.randint(1, len(words) - 1)
            words.insert(insert_pos, filler)
        return ' '.join(words)

    # --- Normalized Fraction Function ---
    def normalize_fraction_text(self, text: str) -> str:
        """
        Normalizes common written fraction patterns like '1 over 4' or '1over4'
        to the standard numeric fraction format '1/4'.
        """
        # We only need to look for digits here since 'word_to_number' runs first.
        # We keep number words in the pattern to ensure robustness against complex inputs.

        pattern = re.compile(
            # Look for either a digit (\d+) OR a number word
            r'(\d+|' + self.fraction_number_words + r')\s*over\s*(\d+|' + self.fraction_number_words + r')',
            re.IGNORECASE
        )

        replacement = r'\1/\2'
        normalized_text = pattern.sub(replacement, text)

        return normalized_text
    # --- NEW METHOD: Word Fraction to Number ---
    def word_fraction_to_number(self, text: str) -> str:
        """
        Converts fractional word phrases (e.g., 'two thirds', 'a quarter')
        to their numerical fraction form (e.g., '2/3', '1/4').
        """
        def replace_fraction(match):
            # Get the matched text and strip to lower case for dictionary lookup
            original_phrase = match.group(0)
            phrase_lower = original_phrase.lower()

            # Get the numerical replacement
            replacement = self.fraction_word_map.get(phrase_lower, original_phrase)
            return replacement

        # Use re.sub with a custom function to replace all occurrences
        text = self.fraction_word_pattern.sub(replace_fraction, text)
        return text
    def clean(self, text: str) -> str:

         # 2. **NEW STEP:** Convert fractional words to digits ("two thirds" -> "2/3")
        text = self.word_fraction_to_number(text)

        # 1. Convert simple number words to digits ("four" -> "4")
        text = self.word_to_number(text)



        # 3. Normalize 'x over y' patterns ("4 over 6" -> "4/6")
        text = self.normalize_fraction_text(text)

        return text


    # --- UPDATED AUGMENT FUNCTION ---
    def augment(self, text: str) -> str:
        """Apply a random subset of augmentations to the text."""
        # 1. First, perform mandatory cleaning steps
        text = self.clean(text)

        # 2. Now, apply random augmentations (typos, synonyms, etc.)
        if random.random() > self.augmentation_prob:
            return text

        augmentations = [
            self.substitute_synonym,
            self.remove_punctuation,
            self.add_filler_words,
            self.randomly_capitalize_word,
            self.apply_typo,
            self.word_fraction_to_number
        ]

        num_augmentations = random.randint(1, 3)
        selected_augmentations = random.sample(augmentations, min(num_augmentations, len(augmentations)))

        for aug_func in selected_augmentations:
            text = aug_func(text)

        return text
def standardize_answer_format_v5(answer_str: str) -> str:
    import re
    s = answer_str.strip()
    
    if s in ["Not enough information", "Unknowable", "Certain", "Likely", "Unlikely",'Impossible','Very Likely']:
        return s

    # 1. FIX THE "1 \frac{4}{9}" CONSISTENCY ISSUE
    # If it has a whole number followed by \frac but NO \( wrapper
    mixed_latex_pattern = re.compile(r'^(\d+)\s+(\\frac\{.*?\}\{.*\})(?:\s+([a-zA-Z]+))?$')
    mixed_latex_match = mixed_latex_pattern.match(s)
    if mixed_latex_match:
        w, f, u = mixed_latex_match.groups()
        res = f'\\( {w} {f} \\)'
        return f"{res} {u}" if u else res

    # 2. ALREADY WRAPPED PROTECTION
    if s.startswith('\\(') and s.endswith('\\)'):
        return s

    # 3. RAW MIXED NUMBER (e.g., "1 3/4")
    mixed_raw_pattern = re.compile(r'^(\d+)\s+(\d+)\s*/\s*(\d+)(?:\s+([a-zA-Z]+))?$')
    mixed_raw_match = mixed_raw_pattern.match(s)
    if mixed_raw_match:
        w, n, d, u = mixed_raw_match.groups()
        res = f'\\( {w} \\frac{{{n}}}{{{d}}} \\)'
        return f"{res} {u}" if u else res

    # 4. SIMPLE FRACTION (e.g., "3/4")
    simple_frac_pattern = re.compile(r'^(-?\s*\d+)\s*/\s*(\d+)(?:\s+([a-zA-Z]+))?$')
    simple_frac_match = simple_frac_pattern.match(s)
    if simple_frac_match:
        n, d, u = simple_frac_match.groups()
        res = f'\\( \\frac{{{n.strip()}}}{{{d}}} \\)'
        return f"{res} {u}" if u else res

    # 5. BARE NUMBERS (e.g., "43")
    # Using a restrictive regex so it doesn't double-wrap stuff
    if not s.startswith('\\('):
        num_match = re.match(r'^([-]?\d+\.?\d*)\s*([a-zA-Z]*)$', s)
        if num_match:
            v, u = num_match.groups()
            res = f'\\( {v} \\)'
            return f"{res} {u}".strip() if u else res

    # 6. FINAL FALLBACK
    if not s.startswith('\\('):
        return f'\\( {s} \\)'
    return s
def strip_extra_whitespace(text_string: str) -> str:
    """
    Removes leading, trailing, and redundant internal whitespace
    (spaces, tabs, newlines) from a string.

    It first splits the string on any whitespace sequence, which
    automatically handles multiple internal spaces, and then joins
    the resulting words back together with a single space.

    Args:
        text_string: The input string possibly containing extra whitespace.

    Returns:
        The cleaned string with all redundant whitespace removed.
    """
    if not text_string:
        return text_string
    #text_string=text_string.replace('\n',' ')
    text_string=re.sub('\n',' ',text_string)
    # 1. Split the string by *any* sequence of whitespace.
    #    This creates a list of "words" while discarding empty strings
    #    (which result from multiple spaces, tabs, or newlines).
    words = text_string.split()

    # 2. Join the list of words back together using a single space.
    cleaned_string = " ".join(words)

    return cleaned_string
def rephrase_question_and_explanation(question_text: str) -> tuple[str, str]:
    """
    Rephrases question text to prevent linguistic bias in training data.
    The explanation is returned unchanged.
    """

    # --- 1. Question Rephrasing Logic ---

    # Determine question type (largest/smallest/greatest/etc)
    question_type = None
    for term in ['largest', 'greatest', 'biggest', 'smallest', 'larger', 'smaller', 'greater', 'lesser']:
        if term in question_text.lower():
            question_type = term
            break

    # Extract numbers from the question
    # Look for "Options:" followed by numbers
    options_match = re.search(r'Options?:?\s*(.+?)(?:\n|$)', question_text)
    if options_match:
        numbers_str = options_match.group(1).strip()
    else:
        # Fallback - try to extract all numbers, though this is less reliable
        # We assume the numbers are after the question mark and comma-separated
        parts = question_text.split('?')
        if len(parts) > 1:
            numbers_str = parts[-1].strip()
        else:
            numbers_str = "Unknown Options"

    # Map question types to their variants
    largest_variants = {
        'largest': ['largest', 'greatest', 'biggest'],
        'greatest': ['largest', 'greatest', 'biggest'],
        'biggest': ['largest', 'greatest', 'biggest'],
        'larger': ['larger', 'greater', 'bigger'],
        'greater': ['larger', 'greater', 'bigger'],
        'bigger': ['larger', 'greater', 'bigger']
    }

    smallest_variants = {
        'smallest': ['smallest', 'least'],
        'smaller': ['smaller', 'lesser'],
        'lesser': ['smaller', 'lesser'],
        'least': ['smallest', 'least']
    }

    # Determine if it's a superlative (largest/smallest) or comparative (larger/smaller)
    is_superlative = question_type in ['largest', 'greatest', 'biggest', 'smallest', 'least']
    is_max_question = question_type in ['largest', 'greatest', 'biggest', 'larger', 'greater', 'bigger']

    # Choose a random variant
    new_type = question_type # Default to original if not found
    if is_max_question:
        if is_superlative:
            # Use 'largest' set of synonyms
            new_type = random.choice(largest_variants.get('largest', ['largest']))
        else:
            # Use 'larger' set of synonyms
            new_type = random.choice(largest_variants.get('larger', ['larger']))
    elif question_type: # Must be a min question
        if is_superlative:
            # Use 'smallest' set of synonyms
            new_type = random.choice(smallest_variants.get('smallest', ['smallest']))
        else:
            # Use 'smaller' set of synonyms
            new_type = random.choice(smallest_variants.get('smaller', ['smaller']))

    # Question stem variants
    if is_superlative:
        question_stems = [
            f"Which number is the {new_type}?",
            f"What number is {new_type}?",
            f"Identify the {new_type} number.",
            f"Choose the {new_type} number.",
            f"Pick the {new_type} number.",
            f"Select the {new_type} number.",
            f"Find the {new_type} number.",
            f"Which is the {new_type}?",
            f"What is the {new_type}?",
            f"Determine the {new_type} number.",
        ]
    else:
        question_stems = [
            f"Which number is {new_type}?",
            f"What number is {new_type}?",
            f"Identify the {new_type} number.",
            f"Choose the {new_type} number.",
            f"Pick the {new_type} number.",
            f"Select the {new_type} number.",
            f"Find the {new_type} number.",
            f"Which is {new_type}?",
            f"What is {new_type}?",
            f"Determine which number is {new_type}.",
        ]

    # If no type was found, return original question
    if not question_type:
        return question_text

    question_stem = random.choice(question_stems)

    # Options formatting variants
    options_formats = [
        f"\n\nOptions: {numbers_str}",
        f"\n\n{numbers_str}",
        f" Options: {numbers_str}",
        f" {numbers_str}",
    ]

    options_format = random.choice(options_formats)

    new_question = question_stem + options_format

    # --- 2. Explanation Logic (Return Unchanged) ---
    if random.random()<0.5:
      if random.random()<0.5:
        new_question=re.sub('Options:',' ',new_question)
      else:
        new_question=re.sub('Options:',' ',new_question)
    if random.random()<0.5:
      if random.random()<0.5:
        new_question=re.sub('\n',' ',new_question)
      else:
        new_question=re.sub('\n',' ',new_question)

    # The explanation is returned as-is, with no modification.
    return new_question
# ============================================================================
# MODEL CLASS
# ============================================================================

class MultiTaskQwen(nn.Module):
    def __init__(self, pretrained_model_name, num_categories, num_misc_classes, bnb_config=None):
        """
        Multi-task model using Qwen and fine-tuning the classification heads.

        Args:
            pretrained_model_name: Name of pretrained transformer model (e.g., 'Qwen/Qwen2-7B-Instruct')
            num_categories: Number of category classes
            num_misc_classes: Number of misconception classes
            bnb_config: BitsAndBytesConfig for 4-bit loading
        """
        super(MultiTaskQwen, self).__init__()

        # Load Qwen as AutoModel
        self.qwen = AutoModel.from_pretrained(
            pretrained_model_name,
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        hidden_size = self.qwen.config.hidden_size

        # Task-specific heads
        self.correctness_head = nn.Linear(hidden_size, 2)
        self.category_head = nn.Linear(hidden_size, num_categories)
        self.misconception_head = nn.Linear(hidden_size, num_misc_classes)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.
        """
        # Process text with Qwen
        outputs = self.qwen(input_ids=input_ids, attention_mask=attention_mask)

        # For CausalLMs (like Qwen), extract the last non-padding token's hidden state
        last_hidden_state = outputs.last_hidden_state

        # Find the last token index that is NOT padding
        sequence_lengths = torch.sum(attention_mask, dim=1) - 1

        # Extract the hidden state for the effective last token
        text_representation = last_hidden_state[
            torch.arange(last_hidden_state.size(0), device=input_ids.device),
            sequence_lengths
        ]

        # Task-specific predictions
        correctness_logits = self.correctness_head(text_representation)
        category_logits = self.category_head(text_representation)
        misconception_logits = self.misconception_head(text_representation)

        return correctness_logits, category_logits, misconception_logits
