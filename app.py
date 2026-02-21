# ==========================================
# ==========================================
# CRITICAL: spaces MUST be imported FIRST
# ==========================================
import os
import sys
print(f"This module has been updated feb 16, {os.getcwd()}")
# Environment detection
IS_ON_SPACES = 'SPACE_ID' in os.environ

# Import spaces BEFORE torch/CUDA
if IS_ON_SPACES:
    try:
        import spaces
        SPACES_AVAILABLE = True
        print("✅ Spaces library loaded")
    except ImportError:
        spaces = None
        SPACES_AVAILABLE = False
        print("⚠️ Spaces library not available")
else:
    spaces = None
    SPACES_AVAILABLE = False

# ==========================================
# CONFIGURATION & PATH SETUP
# ==========================================
if IS_ON_SPACES:
    project_root = '.'
    scripts_dir = './utils'
    models_dir = './models'
else:
    cwd = os.getcwd()
    if "LLM_Misconception_Project" in cwd:
        project_root = cwd
    else:
        project_root = os.path.join(cwd, "LLM_Misconception_Project")
    
    scripts_dir = os.path.join(project_root, "utils")
    models_dir = os.path.join(project_root, "models")

# Add scripts to path BEFORE importing custom modules
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

print(f"📁 Project root: {project_root}")
print(f"📁 Scripts dir: {scripts_dir}")
print(f"📁 Models dir: {models_dir}")

# Import custom modules FIRST (before torch)
import preprocessing_functions
import augment_functions_map41 as augment_functions

print('✅ Setup complete. No extra path variables, no duplicates.')

# ==========================================
# NOW safe to import CUDA-related packages
# ==========================================
import warnings
warnings.filterwarnings('ignore')

import re
import math
import time
import random
import joblib
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_recall_fscore_support

from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from bitsandbytes.optim import AdamW8bit

import gradio as gr
from datetime import datetime
from huggingface_hub import hf_hub_download
from latex2mathml.converter import convert
# Global results cache
results_cache = {'results_df': None}

print('The module has been updated')

# ==========================================
# HELPER FUNCTION: Download models from HF
# ==========================================
def get_model_path2(filename: str, repo_id: str = "jprich1984/math-misconception-models") -> str:
    """
    Download model from HuggingFace Models repo if needed.
    Falls back to local path for Colab compatibility.
    
    Args:
        filename: Name of the checkpoint file
        repo_id: HuggingFace repo ID
    
    Returns:
        Local path to the model file
    """
    # Check if running on HF Spaces
    if 'SPACE_ID' in os.environ:
        print(f"📥 Downloading {filename} from HuggingFace...")
        try:
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir="./model_cache"
            )
            print(f"✅ Downloaded to: {model_path}")
            return model_path
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")
            raise
    else:
        # Running in Colab - use local path
        local_path = os.path.join("checkpoints", filename)
        if os.path.exists(local_path):
            print(f"✅ Using local model: {local_path}")
            return local_path
        else:
            # Try downloading even in Colab
            print(f"📥 Local file not found, downloading {filename}...")
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename
            )
            return model_path

def get_model_path(filename: str, repo_id: str = "jprich1984/math-misconception-models") -> str:
    """
    Download model from HuggingFace Models repo if needed.
    Falls back to local path for Colab compatibility.
    
    Args:
        filename: Name of the checkpoint file
        repo_id: HuggingFace repo ID
    
    Returns:
        Local path to the model file
    """
    # Check if running on HF Spaces
    if 'SPACE_ID' in os.environ:
        print(f"📥 Downloading {filename} from HuggingFace...")
        try:
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir="./model_cache"
            )
            print(f"✅ Downloaded to: {model_path}")
            return model_path
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")
            raise
    else:
        # Running in Colab - use ABSOLUTE path
        # First, determine the project root
        if 'LLM_Misconception_Project' in os.getcwd():
            # Already inside project directory
            project_root = os.getcwd().split('LLM_Misconception_Project')[0] + 'LLM_Misconception_Project'
        else:
            # Assume standard Colab location
            project_root = '/content/LLM_Misconception_Project'
        
        local_path = os.path.join(project_root, "checkpoints", filename)
        
        if os.path.exists(local_path):
            print(f"✅ Using local model: {local_path}")
            return local_path
        else:
            # File doesn't exist locally - try downloading
            print(f"⚠️ Local file not found at: {local_path}")
            print(f"📥 Attempting to download {filename} from HuggingFace...")
            return hf_hub_download(repo_id=repo_id, filename=filename)


def generate_assessment_questions(df,specific_class=False):
  df=df.sample(frac=1)
  df['Primary_Misconception']=df['Misconception'].apply(lambda x: x[0])
  grouped_df=df.groupby('Primary_Misconception').agg({'QuestionText':'first','problem_type':'first','Primary_Misconception':'first','Misconception':'first'})
  if not specific_class:
    grouped_df=df.groupby('problem_type').first().reset_index()
  else:
    grouped_df=df.sample(2)
  return grouped_df

def extract_misconception_values(row):
    """
    Comprehensive extractor for misconception distractors across all 15 types.
    Includes categorical overrides, geometry padding, and decimal extraction.
    """
    mapping_dict = get_mapping_dict()
    pt_id = row['problem_type']
    context_str = str(row.get('Mathematical_Context', ""))

    # --- 1. FIXED OVERRIDES & SPECIAL LOGIC ---

    # Type 10: Probability Word Labels (Judgment based)
    if pt_id == 10:
        return ['Likely', 'Unlikely', 'Impossible', 'Certain']

    # Type 6: Decimal Comparison (Extracting the actual numbers being compared)
    if pt_id == 6:
        # Pulls values like 4.4, 4.004 from the raw context string
        return list(dict.fromkeys(re.findall(r"[-+]?\d*\.\d+|\d+", context_str)))

    # --- 2. DYNAMIC MAPPING EXTRACTION ---

    found_values = []
    pt_mapping = mapping_dict.get(pt_id)

    if pt_mapping:
        # Flatten all target keys we are looking for (e.g., 'RATIO_SIM', 'N1+N2')
        all_target_keys = []
        for keys in pt_mapping.values():
            if isinstance(keys, list):
                all_target_keys.extend(keys)
            elif isinstance(keys, str):
                all_target_keys.append(keys)

        # Parse context: "KEY:VAL | KEY:VAL"
        parts = [p.strip() for p in context_str.split('|') if p.strip()]
        for part in parts:
            if ':' in part:
                key, val = part.split(':', 1)
                if key.strip() in all_target_keys:
                    found_values.append(val.strip())

    # --- 3. SPECIFIC PADDING FOR GEOMETRY (Type 11) ---

    if pt_id == 11 and found_values:
        expanded_values = set(found_values)
        for val in found_values:
            try:
                # Convert to float then int to handle strings like "6.0"
                num = int(float(val))
                # Add neighborhood distractors (catching off-by-one or n-2 errors)
                expanded_values.add(str(num - 1))
                expanded_values.add(str(num + 1))
                expanded_values.add(str(num + 2))
            except ValueError:
                continue
        # Clean up and sort numerically
        return sorted(list(expanded_values), key=lambda x: float(x) if x.replace('.','').replace('-','').isdigit() else 0)

    if pt_id == 12 and found_values:
        expanded_values = set(found_values)
        for val in found_values:
            try:
                # Convert to float then int to handle strings like "6.0"
                num = int(float(val))
                # Add neighborhood distractors (catching off-by-one or n-2 errors)
                expanded_values.add(str(-1*int(num)))
            except ValueError:
                continue
        # Clean up and sort numerically
        return sorted(list(expanded_values), key=lambda x: float(x) if x.replace('.','').replace('-','').isdigit() else 0)

    # --- 4. SAFETY FALLBACK ---

    # If the list is still empty (and it's not Type 10), grab any number found in context
    if not found_values:
        found_values = re.findall(r"[-+]?\d*\.?\d+", context_str)

    # --- 5. CONVERT MIXED FRACTIONS TO LATEX ---
    # Pattern: 2_3/5 → \( 2 \frac{3}{5} \)
    converted_values = []
    mixed_fraction_pattern = re.compile(r'^(-?\d+)_(\d+)/(\d+)$')

    for val in found_values:
        match = mixed_fraction_pattern.match(val)
        if match:
            whole = match.group(1)
            numerator = match.group(2)
            denominator = match.group(3)
            latex_mixed = f"{whole} {numerator}/{denominator}"
            converted_values.append(latex_mixed)
        else:
            converted_values.append(val)

    # Deduplicate while preserving order
    return list(dict.fromkeys(converted_values))

def get_mapping_dict():
  mc_mapping = {
      1: {
          'Incorrect_A': ["F1xN3_SwapDividend_UNSIM"],
          'Incorrect_B': ["F1xN3_SwapDividend_MIXED", "FlipChange_UNSIM_MIXED"],
          'Incorrect_C': ['FlipChange_SIM_MIXED'],
          'Incorrect_D': ["FlipChange_UNSIM2"],
          'Incorrect_E': ["FlipChange_SIM2"],
          'Correct_Unsimp': ["CorrectAnswer_UNSIM_MIXED"],
          'Correct_Simp': ["CorrectAnswer_SIM_MIXED", "F1divN3_SIM"]
      },
      2: {
          'Correct_Simp': ['CORRECT_ANSWER_SIMPLIFIED'],
          'Correct_Unsimp': ['CORRECT_ANSWER'],
          'Incorrect_A': ["FIRST_TERM_DUPLICATION_UNSIM"],
          'Incorrect_B': ["FIRST_TERM_DUPLICATION_SIM"],
          'Incorrect_C': ['FIRST_TERM_INVERSION_UNSIM', 'FIRST_TERM_INVERSION_SIM'],
          'Incorrect_D': ['MISCONCEPTION_FIRST_TERM_TIMES_N'],
          'Incorrect_E': ['MISCONCEPTION_FIRST_TERM_TIMES_N_SIMPLIFIED'],
          'Incorrect_F': [f'WRONG_TERM_{i}' for i in range(11)],
          'Incorrect_G': ['COMMON_DIFF_DUPLICATION_UNSIM', 'COMMON_DIFF_DUPLICATION_SIM'],
          'Incorrect_H': ['COMMON_DIFF_INVERSION_UNSIM', 'COMMON_DIFF_INVERSION_SIM']
      },
      3: {
          'Correct_Simp': ['F1xN3_CORRECT_SIM', 'F1xN3_CORRECT_MIXED','CORRECT_ANSWER_SIMP',f"CORRECT_ANSWER_SIMP_MIXED"],





          'Correct_Unsimp': ['F1xN3_CORRECT_UNSIM','CORRECT_ANSWER_UNSIMP',f"CORRECT_ANSWER_UNSIMP_MIXED"],
          'Incorrect_A': ['F1divN3_UNSIM', 'F1divN3_SIM'],
          'Incorrect_B': ['DuplicationAnswer_UNSIM', 'DuplicationAnswer_SIM',f"DuplicationAnswer_UNSIM_MIXED",f"DuplicationAnswer_SIM_MIXED"],
          'Incorrect_C': ['InversionAnswer_UNSIM', 'InversionAnswer_SIM',f"InversionAnswer_UNSIM_MIXED",
f"InversionAnswer_SIM_MIXED"],
          'Incorrect_D': ['WRONG_OPERATION_ANSWER_UNSIMP', 'WRONG_OPERATION_ANSWER_SIMP'],
          'Incorrect_E': ['F1_BOTH_MULT_N3'],
          'Incorrect_F': ['N3divF1_SIM'],
          'Incorrect_G': ['F1xN3_UNSIM']
      },
      4: {
          'Correct_Unsimp': [
              'COMP_NUM_TIMES_WHOLE_DIV_DEN_UNSIM',
              'NUM_TIMES_WHOLE_DIV_DEN_UNSIM'
          ],
          'Incorrect_Simp': [
              'COMP_NUM_TIMES_WHOLE_DIV_DEN_SIMP',
              'NUM_TIMES_WHOLE_DIV_DEN_SIMP'
          ],
          'Incorrect_A': [
              'WHOLE_DIV_DEN',
              'WHOLE_DIV_DEN_SIMP',
              'COMP_WHOLE_DIV_DEN',
              'COMP_WHOLE_DIV_DEN_SIMP'
          ],
          'Incorrect_B': [
              'WHOLE_DIV_NUM'
          ],
          'Incorrect_C': [
              'COMP_NUM_TIMES_WHOLE'
          ],
          'Incorrect_D': [
              'COMP_DEN_TIMES_WHOLE'
          ]
      },
      5: {
          'Correct_Simp': [
              'F1+F2_CORRECT_SIM'
          ],
          'Correct_Unsimp': [
              'F1+F2_CORRECT_UNSIM'
          ],
          'Incorrect_A': [
              'F1+F2_ADDING_ACROSS',
              'F1+F2_ADDING_ACROSS_SIM',

          ],
          'Incorrect_B': [
              'F1+F2_DENOM_ONLY_CHANGE',
              'F1+F2_DENOM_ONLY_CHANGE_SIM'
          ],
          'Incorrect_C': [
              'F1+F2_DOUBLED_DENOM_UNSIM',
              'F1+F2_DOUBLED_DENOM_SIM'
          ],
          'Incorrect_D': [
              'F1+F2_USING_FIRST_DEN',
              'F1+F2_USING_SECOND_DEN'
          ],
          'Incorrect_E': [
              'F1+F2_JUST_ADD_NUM_FIRST_DEN',
              'F1+F2_JUST_ADD_NUM_SECOND_DEN'
          ]},



      7:{'All':[f'N{i}' for i in range(4)]},

      8: {
          'Correct_Simp': [
              'CORRECT_ANSWER'              # 69.25
          ],
          'Incorrect_G': [
              'UNIT_RATE'                   # 92.3333... (Commonly a middle step)
          ],
          'Incorrect_A': [
              'MULT_BY_MULT_ANSWER'         # 1108 (Multiplying instead of dividing)
          ],
          'Incorrect_B': [
              'N1xN2',                      # 36 (Operational confusion: N1 and N2 only)
              'N1+N2'                       # 15
          ],
          'Incorrect_C': [
              'N2xN3',                      # 3324
              'N2+N3'                       # 289
          ],
          'Incorrect_D': [
              'N1xN3',                      # 831
              'N1+N3'                       # 280
          ],
          'Incorrect_E': [
              'N3divN1',                    # 92.3333
              'N3divN2'                     # 23.0833
          ],
          'Incorrect_F': [
              'MULTIPLIER'                  # 4 (Student identifies the scale but not the value)
          ]
      },

      # ... (Entries 1-8 as before)
      9: {
          'Correct_Simp': [
              'F1_TIMES_F2_SIMPLIFIED'        # 5/18
          ],
          'Correct_Unsimp': [
              'F1_TIMES_F2'                   # 5/18 (already simplified here)
          ],
          'Incorrect_A': [
              'F1_DIV_F2_SIMPLIFIED',         # 5/2
                    # 2/5
          ],
          'Incorrect_B': [
              'F1_DIV_F2',                    # 15/6
              'F2_DIV_F1'                     # 6/15
          ],
          'Incorrect_C': [
              'F1_SUBTRACT_F2_SIMPLIFIED',    # 1/2
                # -1/2
          ],
          'Incorrect_D': [
              'F1_SUBTRACT_F2',                # 3/6
                      # -3/6
          ]
      },
      10:None,
      11: {
          'Correct_Simp': [
              'N1_SIDES_FROM_EXT'  # 6 (Logic: 360 / 60)
          ],
          'Incorrect_A': [
              'N1_SIDES_FROM_INT' ] # 3 (Logic: 180 * (n-2) / n = 60)

      },
      12: {
          'Correct_Simp': ['CORRECT_ANSWER'],
          'Incorrect_A': ['N1_N2_ABS_DIFF'],
          'Incorrect_B':['N1_N2_ABS_SUM'],
          'Incorrect_C':['N1+N2'],
          'Incorrect_D':['N1-N2']

      },
      13:{'Correct_Simp':['CORRECT_OPERATION_MULTIPLY','CORRECT_OPERATION_ADD','CORRECT_OPERATION_SUBTRACT'],
          'Incorrect_A':['WRONG_OPERATION_MULTIPLY','WRONG_OPERATION_ADD','WRONG_OPERATION_SUBTRACT'],
          'Incorrect_B':['EQ_TAIL_DIGITS_MISC'],
          'Incorrect_C':['N1+N2'],
          'Incorrect_D': ['N1-N2'],
          'Incorrect_E': ['N1xN2']
      },
      14: {'Correct_Simp':['CORRECT_ANSWER_SIM'],
          'Correct_Unsimp':['CORRECT_ANSWER_UNSIM'],
          'Incorrect_A':['ADDITIVE_ANSWER'],
          'Incorrect_B':['F1_BOTH_MULT_N2'],
          'Incorrect_C': ['F1divN2_UNSIM'],
          'Incorrect_D': ['F1divN2_SIM'],
          'Incorrect_E': ['N2divF1_SIM'],
          'Incorrect_F': ['F1xN3_UNSIM'],
          'Incorrect_G':['F1xN3_SIM']
          },
      15: {
          'Correct_Simp': ['N1_DIVIDE_TOTAL_SIM', 'COMP_ANSWER_SIM'],
          'Correct_Unsimp': ['N1_DIVIDE_TOTAL_UNSIM', 'COMP_ANSWER_UNSIM'],
          'Incorrect_A': ['RATIO_SIM'],       # Part-to-part logic (3:12 -> 1/4)
          'Incorrect_B': ['RATIO_UNSIM'],     # Part-to-part logic (3/12)
          'Incorrect_C': ['RATIO_TWO_SIM'],   # Inverse ratio (12:3 -> 4)
  # Just the count of unshaded (12)
          # Just the total count (15)
          'Incorrect_G': ['RATIO_TWO_UNSIM']  # Unsimplified inverse ratio (12/3)
      }




  }
  return mc_mapping


def add_dual_model_predictions(grouped,
                               correctness_model,
                               misconception_model,
                               tokenizer,
                               label_encoder_category,
                               label_encoder_misconception,
                               device,
                               batch_size=16,
                               max_length=450,
                               add_ground_truth=False):
    """
    Add predictions using TWO models:
    - Correctness model (with full mathematical context)
    - Misconception model (with reduced boolean context)
    - Category derived from correctness + misconception rules
    """
    correctness_model.eval()
    misconception_model.eval()

    # Storage
    all_correctness_preds = []
    all_category_preds = []
    all_misconception_preds = []
    all_misconception_probs = []

    print(f"Running dual-model inference on {len(grouped)} samples...")

    with torch.no_grad():
        for i in range(0, len(grouped), batch_size):
            batch_df = grouped.iloc[i:i+batch_size]

            # ========== PREPARE INPUTS ==========
            questions_with_answers = []
            explanations_full = []
            explanations_reduced = []

            for _, row in batch_df.iterrows():
                question_with_answer = f"{row['QuestionText']} {row['MC_Answer']}"
                questions_with_answers.append(question_with_answer)

                # Full context for correctness
                math_context_full = format_math_context_as_text(row['Mathematical_Context'])
                explanations_full.append(math_context_full + " " + row['StudentExplanation'])

                # Reduced context for misconception
                math_context_reduced = format_math_context_as_text(row['Mathematical_Context_Reduced'])
                explanations_reduced.append(math_context_reduced + " " + row['StudentExplanation'])

            # ========== CORRECTNESS PREDICTION (Full Features) ==========
            encoding_full = tokenizer(
                questions_with_answers,
                explanations_full,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )

            input_ids_full = encoding_full['input_ids'].to(device)
            attention_mask_full = encoding_full['attention_mask'].to(device)

            logits_corr, _, _ = correctness_model(input_ids_full, attention_mask_full)
            preds_correctness = logits_corr.argmax(dim=-1).cpu().numpy()

            # ========== MISCONCEPTION PREDICTION (Reduced Features) ==========
            encoding_reduced = tokenizer(
                questions_with_answers,
                explanations_reduced,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )

            input_ids_reduced = encoding_reduced['input_ids'].to(device)
            attention_mask_reduced = encoding_reduced['attention_mask'].to(device)

            _, _, logits_misc = misconception_model(input_ids_reduced, attention_mask_reduced)
            misconception_probs = torch.sigmoid(logits_misc).cpu().numpy()
            preds_misconception = (misconception_probs > 0.5).astype(int)

            # ========== DERIVE CATEGORY FROM RULES ==========
            batch_size_actual = len(preds_correctness)
            preds_category = []

            for j in range(batch_size_actual):
                # Get predicted misconception (highest probability)
                misc_idx = np.argmax(misconception_probs[j])
                misc_name = label_encoder_misconception.inverse_transform([misc_idx])[0]

                is_correct = preds_correctness[j] == 1

                # Apply category rules
                if misc_name not in ['No Misconception', 'Other']:
                    cat_name = 'True_Misconception' if is_correct else 'False_Misconception'
                elif misc_name == 'No Misconception':
                    cat_name = 'True_Correct' if is_correct else 'False_Correct'
                else:  # Other
                    cat_name = 'True_Neither' if is_correct else 'False_Neither'

                preds_category.append(cat_name)

            # Store predictions
            all_correctness_preds.extend(preds_correctness)
            all_category_preds.extend(preds_category)
            all_misconception_preds.extend(preds_misconception)
            all_misconception_probs.extend(misconception_probs)

            if ((i // batch_size) + 1) % 10 == 0:
                print(f"  Processed {i + len(batch_df)}/{len(grouped)} samples")

    # Convert predictions to labels
    correctness_labels = ['Incorrect' if p == 0 else 'Correct' for p in all_correctness_preds]

    # Get misconception names (all above threshold)
    misconception_labels = []
    for pred in all_misconception_preds:
        misconceptions = [label_encoder_misconception.classes_[i]
                         for i, val in enumerate(pred) if val == 1]
        misconception_labels.append(misconceptions if misconceptions else ['None'])

    # Get top misconception (highest probability)
    top_misconception_labels = []
    for probs in all_misconception_probs:
        top_idx = np.argmax(probs)
        top_misc = label_encoder_misconception.inverse_transform([top_idx])[0]
        top_misconception_labels.append(top_misc)

    # Add to dataframe
    grouped = grouped.copy()
    grouped['predicted_correctness'] = correctness_labels
    grouped['predicted_category'] = all_category_preds
    grouped['predicted_misconception_all'] = misconception_labels
    grouped['predicted_misconception_top'] = top_misconception_labels

    print(f"\n✅ Predictions added to dataframe")
    print(f"   Correctness distribution: {pd.Series(correctness_labels).value_counts().to_dict()}")
    print(f"   Category distribution: {pd.Series(all_category_preds).value_counts().to_dict()}")

    return grouped


def format_math_context_as_text(context_string):
    """Convert math context to text format"""
    if not context_string or not isinstance(context_string, str):
        return ""

    parts = [part.strip() for part in context_string.split('|') if part.strip()]
    formatted_features = []

    for part in parts:
        if ':' in part:
            key, value = part.split(':', 1)
            formatted_features.append(f"{key.strip()}:{value.strip()}")

    if not formatted_features:
        return ""

    return " <MATH_CONTEXT> " + " | ".join(formatted_features) + " </MATH_CONTEXT> "



def display_final_results(prediction_df,report_for='educator'):
  for ind, row in prediction_df.iterrows():
    problem_type=row['problem_type']
    for misconception in row['predicted_misconception_all']:
      if misconception not in [None,'None','none','No Misconception']:
        print()
        if report_for=='educator':
          results_dict=get_recommendations_educator(misconception,problem_type)
        else:
          results_dict=get_recommendations(misconception,problem_type)
        results_text=results_dict['text']
        results_links=results_dict['links']
        print(f'Detected Misconception: {misconception}')
        print(f'\tQuestion: {row['QuestionText']}')
        print(f'\t{results_text}')
        print(f"\treccommended links: {results_links}")

def get_recommendations(misconception,problem_type):

  general_recommendation_map = {
      1: {
          'text': 'Focus on the relationship between division and multiplication. Remember, dividing by a fraction is the same as multiplying by its reciprocal (flipping the second number)!',
          'links': [
              'https://www.youtube.com/watch?v=4lkq3DgvmGw',
              'https://www.youtube.com/watch?v=fpx7XkG0O3s'
          ]
      },
      2: {
          'text': 'When finding the next step in a pattern, identify the "common difference." Don’t just multiply the first number; track how much the pattern grows at every single step.',
          'links': [
              'https://www.youtube.com/watch?v=V7m8m6U2u4M',
              'https://www.youtube.com/watch?v=XpG-9T_7j0E'
          ]
      },
      3: {
          'text': 'Multiplying fractions is like finding a "part of a part." Multiply the numerators (top) together and the denominators (bottom) together—you do not need a common denominator!',
          'links': [
              'https://www.youtube.com/watch?v=Rb4XW_V0Uvc',
              'https://www.youtube.com/watch?v=qM2oW9_T7X0'
          ]
      },
      4: {
          'text': 'Read carefully to see if you need the amount mentioned or the "leftover" (remainder) amount. Drawing a bar model can help you see which part of the whole you are looking for.',
          'links': [
              'https://www.youtube.com/watch?v=7uV_G_Tf_YI',
              'https://www.youtube.com/watch?v=cbV83Xv6Kls'
          ]
      },
      5: {
          'text': 'You must have a "Common Denominator" before adding or subtracting. This ensures the slices you are combining are the exact same size before you total them up.',
          'links': [
              'https://www.youtube.com/watch?v=bcCLlzS_6Yc',
              'https://www.youtube.com/watch?v=N-Y0KvgS6no'
          ]
      },
      7: {
          'text': 'Decimal value isn’t determined by the length of the number. Compare the place value columns (Tenths, Hundredths) starting from the left to see which value is truly larger.',
          'links': [
              'https://www.youtube.com/watch?v=cbV83Xv6Kls',
              'https://www.youtube.com/watch?v=KzfWUEJjG18'
          ]
      },
      8: {
          'text': 'In work problems, more people means less time! Find the total "work" (People x Time) first, then divide that total by the new number of workers to find the new time.',
          'links': [
              'https://www.youtube.com/watch?v=H7mYVvXGisQ',
              'https://www.youtube.com/watch?v=52Yf7_W_C-M'
          ]
      },
      9: {
          'text': 'When reducing or sharing amounts, ensure you aren’t just subtracting. Division helps us find how many equal parts a total can be split into.',
          'links': [
              'https://www.youtube.com/watch?v=vV38D_r7Y_E',
              'https://www.youtube.com/watch?v=qM2oW9_T7X0'
          ]
      },
      10: {
          'text': 'All probabilities exist on a scale from 0 (Impossible) to 1 (Certain). A decimal like 0.9 is very close to 1, meaning the event is highly likely to happen!',
          'links': [
              'https://www.youtube.com/watch?v=cbV83Xv6Kls',
              'https://www.youtube.com/watch?v=KzfWUEJjG18'
          ]
      },
      11: {
          'text': 'Every regular polygon follows specific rules for its angles. Use the exterior angle (180 - interior) and the "360 rule" to calculate the number of sides.',
          'links': [
              'https://www.youtube.com/watch?v=fS6uV_XmS9Y',
              'https://www.youtube.com/watch?v=m7X3tK9S_Sg'
          ]
      },
      12: {
          'text': 'When working with negative numbers, imagine a number line. Subtracting a negative is the same as adding a positive—it moves your position to the right!',
          'links': [
              'https://www.youtube.com/watch?v=qM2oW9_T7X0',
              'https://www.youtube.com/watch?v=C38B33ZywWs'
          ]
      },
      13: {
          'text': 'Keep variables and constants separate. You can only combine "like terms." If solving for a variable, always perform the inverse operation to both sides of the equation.',
          'links': [
              'https://www.youtube.com/watch?v=CLWpkv6S7_Y',
              'https://www.youtube.com/watch?v=C_B_C8f5f5o'
          ]
      },
      14: {
          'text': 'To keep a fraction equivalent, you must multiply or divide the top and bottom by the same number. Adding or subtracting from the parts changes the ratio!',
          'links': [
              'https://www.youtube.com/watch?v=qcS_S9_kC90',
              'https://www.youtube.com/watch?v=N-Y0KvgS6no'
          ]
      },
      15: {
          'text': 'A fraction is a part out of a total. When looking at a shape, the bottom number (denominator) must represent every single piece, not just the unshaded ones.',
          'links': [
              'https://www.youtube.com/watch?v=n0FZhQ_GkKw',
              'https://www.youtube.com/watch?v=3m693-hItp8'
          ]
      }
  }
  reccomendations_map={'Incomplete': {'text': 'Make sure you simplify your results when necessary','links':['https://www.youtube.com/watch?v=aNQXhknSwrI']},
                      'Longer_is_bigger': {'text': 'Just because a number has more decimal places doesn\'t mean it is larger','links':['https://www.youtube.com/watch?v=RHUl4kZDD6c','https://www.youtube.com/watch?v=trTS_KfkqtI'] },
                      'Shorter_is_bigger': {'text': 'Just because a number has less decimal places doesn\'t mean it is larger','links':['https://www.youtube.com/watch?v=RHUl4kZDD6c','https://www.youtube.com/watch?v=trTS_KfkqtI']},
                      'Ignores_zeroes': {
      'text': 'A zero between the decimal point and a digit (like the 0 in 8.07) acts as a placeholder that changes the value of the numbers that follow it.',
      'links': [
          'https://www.youtube.com/watch?v=trTS_KfkqtI', # Focuses on lining up place values
          'https://www.youtube.com/watch?v=RHUl4kZDD6c'  # Uses the alligator method to compare place-by-place
      ]
  },
    'Whole_numbers_larger': {
      'text': 'A number with a decimal isn’t automatically smaller or "weaker" than a whole number. To compare them fairly, look at the digits to the left of the decimal point first. If those are the same, the decimal part actually makes the number slightly larger than the whole number on its own!',
      'links': [
          'https://www.youtube.com/watch?v=0p_G69v4p9Y', # Comparing whole numbers and decimals
          'https://www.youtube.com/watch?v=RHUl4kZDD6c'  # Place value comparison
      ]
  },
      'Wrong_term': {
      'text': 'It looks like you found the rule for the pattern, but stopped calculating too early. Make sure to apply the rule until you reach the specific pattern number the question is asking for.',
      'links': [
          'https://www.youtube.com/watch?v=XpG-9T_7j0E', # Identifying and extending patterns
          'https://www.youtube.com/watch?v=V7m8m6U2u4M'  # Finding the nth term of a sequence
      ]
  } ,
                      'Duplication': {
      'text': 'When multiplying a fraction by a whole number, you only multiply the numerator (the top number). Multiplying both the top and bottom is like finding an equivalent fraction; it doesn\'t actually change the total value!',
      'links': [
          'https://www.youtube.com/watch?v=Rb4XW_V0Uvc', # Visualizing fraction x whole number
          'https://www.youtube.com/watch?v=17oG76m_36k'  # Why we only multiply the top
      ]
  }    ,
                      'Division': {
      'text': 'When you need to find a "fraction of a fraction" (like 1/2 of 1/4), you should multiply the fractions together, not divide them. Think of the word "of" as a signal to multiply!',
      'links': [
          'https://www.youtube.com/watch?v=qM2oW9_T7X0', # Multiplying fractions in word problems
          'https://www.youtube.com/watch?v=vV38D_r7Y_E'  # Fraction of a fraction visual
      ]
  } ,
                      'Mult': {
      'text': 'It looks like you multiplied the numbers instead of dividing them. When you divide a fraction by a whole number, you are breaking that small piece into even smaller parts, so your answer should be smaller than what you started with!',
      'links': [
          'https://www.youtube.com/watch?v=4lkq3Dgvm74', # Dividing fractions by whole numbers
          'https://www.youtube.com/watch?v=K3f6N-o4SAs'  # Visualizing fraction division
      ]
  } ,
                      'Certainty': {
      'text': 'Probability is a scale from 0 to 1. Only a 1 is truly "Certain" and only a 0 is truly "Impossible." For numbers in between, we use terms like "Likely," "Unlikely," or "Even Chance" to describe the risk or chance involved.',
      'links': [
          'https://www.youtube.com/watch?v=KzfWUEJjG18', # Probability scale and terminology
          'https://www.youtube.com/watch?v=AY3O_SGVtyE'  # Understanding likelihood
      ]
  },
                      'Scale': {
      'text': 'In probability, we only use numbers between 0 and 1. A number like 0.75 or 0.95 might look small, but because the highest possible score is 1, these are actually very high chances! Anything over 0.5 is "Likely," not "Unlikely."',
      'links': [
          'https://www.youtube.com/watch?v=KzfWUEJjG18', # Probability scale: 0 to 1
          'https://www.youtube.com/watch?v=cbV83Xv6Kls'  # Understanding decimals as chances
      ]
  },
                      'Adding_terms': {
      'text': 'When a number and a letter are written right next to each other (like 7w), it actually means they are being multiplied, not added! To solve for the letter, you need to do the opposite of multiplication, which is division.',
      'links': [
          'https://www.youtube.com/watch?v=-_XG9H_mI3U', # Intro to coefficients and multiplication in algebra
          'https://www.youtube.com/watch?v=l3XzepN03KQ'  # Solving one-step equations (multiplication)
      ]
  },
                      'Denominator-only_change': {
      'text': 'When you change the denominator to find a common ground, you must also change the numerator by the same amount. If you multiply the bottom by 3 to get 18, you have to multiply the top by 3 as well to keep the fraction equivalent!',
      'links': [
          'https://www.youtube.com/watch?v=bcCLlzS_6Yc', # Finding common denominators correctly
          'https://www.youtube.com/watch?v=N-Y0KvgS6no'  # Equivalent fractions visual
      ]
  },
                      'Inverse_operation': {
      'text': 'To solve for a variable, you have to "undo" the math by using the opposite operation. If you see multiplication, use division; if you see division, use multiplication. Similarly, if you see addition, use subtraction; and if you see subtraction, use addition!',
      'links': [
          'https://www.youtube.com/watch?v=Qyd_v3DGzTM', # The concept of "Opposite Operations"
          'https://www.youtube.com/watch?v=l3XzepN03KQ'  # One-step equation examples
      ]
  },
                      'Not_variable': {
      'text': 'A letter in an equation (like the "z" in 6z = 66) represents a missing number, not just a digit from the answer! To find its value, you have to use division to see how many times the first number fits into the second one.',
      'links': [
          'https://www.youtube.com/watch?v=vDqOoVux9Zg', # What is a variable?
          'https://www.youtube.com/watch?v=l3XzepN03KQ'  # Solving one-step equations
      ]
  },
                      'Adding_across': {
      'text': 'You cannot add the bottom numbers (denominators) together! Fractions are like slices of a pizza; the bottom number tells you how big the slices are. To add them, the slices must be the same size first. Find a common denominator, and then only add the top numbers.',
      'links': [
          'https://www.youtube.com/watch?v=bcCLlzS_6Yc', # How to add fractions with different denominators
          'https://www.youtube.com/watch?v=N-Y0KvgS6no'  # Visualizing why adding across is wrong
      ]
  },
                      'Subtraction': {
      'text': 'When you see a problem asking for a "fraction of a fraction" (like 1/3 OF 5/6), it actually requires multiplication, even if someone is "giving away" or "eating" a piece. Multiplication helps you find the size of that specific slice relative to the whole!',
      'links': [
          'https://www.youtube.com/watch?v=qM2oW9_T7X0', # Multiplying fractions in word problems
          'https://www.youtube.com/watch?v=vV38D_r7Y_E'  # Finding a fraction of a fraction visually
      ]
  },
                      'Firstterm': {
      'text': 'You can’t just multiply the first number by the pattern number! Patterns usually grow by adding the same amount each step. To find a later pattern, you need to find the "jump" (difference) between the terms and keep adding it until you reach your target.',
      'links': [
          'https://www.youtube.com/watch?v=XpG-9T_7j0E', # Extending arithmetic sequences
          'https://www.youtube.com/watch?v=V7m8m6U2u4M'  # Finding the nth term (a + (n-1)d)
      ]
  },
                      'Interior': {
      'text': 'To find the number of sides from an interior angle, it is usually easiest to find the exterior angle first! Subtract the interior angle from 180, then divide 360 by that result. Remember: the sum of interior angles changes with every side added, but exterior angles always sum to 360.',
      'links': [
          'https://www.youtube.com/watch?v=fS6uV_XmS9Y', # Interior and Exterior angles of polygons
          'https://www.youtube.com/watch?v=m7X3tK9S_Sg'  # Finding the number of sides
      ]
  },
                      'Unknowable': {
      'text': 'Because it is a "regular" polygon, every single interior angle must be the same! This means one angle is actually all you need. You can use that one number to find the exterior angle, and then divide 360 by that to find the number of sides.',
      'links': [
          'https://www.youtube.com/watch?v=fS6uV_XmS9Y', # Properties of Regular Polygons
          'https://www.youtube.com/watch?v=m7X3tK9S_Sg'  # Solving for sides with one angle
      ]
  },
                      'Inversion': {
      'text': 'When multiplying a fraction by a whole number, the whole number should only multiply the numerator (the top). Multiplying the denominator actually divides the fraction and makes it smaller!',
      'links': [
          'https://www.youtube.com/watch?v=Rb4XW_V0Uvc', # Visualizing fraction x whole number
          'https://www.youtube.com/watch?v=17oG76m_36k'  # Why we only multiply the top
      ]
  },
                      'Positive': {
      'text': 'It looks like you turned both numbers into positives! While it is true that "minus a negative" becomes a plus, the first number in your expression stays negative. You should keep the first sign and only change the double negative in the middle.',
      'links': [
          'https://www.youtube.com/watch?v=qM2oW9_T7X0', # Adding and Subtracting Integers
          'https://www.youtube.com/watch?v=C38B33ZywWs'  # Visualizing the number line for negatives
      ]
  },
                      'Scale': {
      'text': 'In probability, all numbers sit between 0 and 1. While 0.99 looks like a small decimal, it is actually the highest possible chance before becoming Certain! Think of it as 99 out of 100—it is almost guaranteed to happen.',
      'links': [
          'https://www.youtube.com/watch?v=cbV83Xv6Kls', # Understanding Decimals as Probabilities
          'https://www.youtube.com/watch?v=KzfWUEJjG18'  # The Probability Scale 0 to 1
      ]
  },
                      'Additive': {
      'text': 'Equivalent fractions are made by multiplying or dividing, never by adding or subtracting! If the denominator is 5 times larger (like 2 becoming 10), the numerator must also be 5 times larger to keep the fraction balanced.',
      'links': [
          'https://www.youtube.com/watch?v=N-Y0KvgS6no', # Understanding Equivalent Fractions
          'https://www.youtube.com/watch?v=qcS_S9_kC90'  # Scaling fractions correctly
      ]
  },
                      'Base_rate': {
      'text': 'To solve this, first find the total "man-hours" (or man-seconds) it takes to finish the job by multiplying the people by the time. Once you know the total work needed, divide that by the new number of employees to see how fast they can finish it together!',
      'links': [
          'https://www.youtube.com/watch?v=H7mYVvXGisQ', # Inverse proportion and work problems
          'https://www.youtube.com/watch?v=52Yf7_W_C-M'  # Man-hours explained
      ]
  },
                      'WNB': {
      'text': 'A fraction should always show the "part" over the "whole" (the total number of pieces). It looks like you put the shaded pieces over the unshaded pieces instead! Try counting every single piece in the shape to find your bottom number (denominator).',
      'links': [
          'https://www.youtube.com/watch?v=n0FZhQ_GkKw', # Basics of fractions: Part of a whole
          'https://www.youtube.com/watch?v=3m693-hItp8'  # Representing fractions with shapes
      ]
  },
                      'Multiplying_by_4': {
      'text': 'To solve this, first find the total "man-hours" (or man-seconds) it takes to finish the job by multiplying the people by the time. Once you know the total work needed, divide that by the new number of employees to see how fast they can finish it together!',
      'links': [
          'https://www.youtube.com/watch?v=H7mYVvXGisQ', # Inverse proportion and work problems
          'https://www.youtube.com/watch?v=52Yf7_W_C-M'  # Man-hours explained
      ]
  },
                      'Tacking': {
      'text': 'Negative signs aren’t just "labels" you can add at the end! They change the direction you move on a number line. To solve these, use the "Keep-Change-Change" rule: Keep the first number, Change minus to plus, and Change the sign of the second number. Then, see where you land!',
      'links': [
          'https://www.youtube.com/watch?v=C38B33ZywWs', # Number line movements for integers
          'https://www.youtube.com/watch?v=qM2oW9_T7X0'  # Rules for subtracting negatives
      ]
  },
                      'Definition': {
      'text': 'The word "polygon" just means a flat shape with straight sides—it doesn’t have a fixed number of sides! To find the exact number of sides for a regular polygon, you have to use the angle provided. A great trick is to subtract the interior angle from 180 to find the exterior angle, then divide 360 by that number.',
      'links': [
          'https://www.youtube.com/watch?v=fS6uV_XmS9Y', # Interior and Exterior angles of polygons
          'https://www.youtube.com/watch?v=m7X3tK9S_Sg'  # Finding the number of sides
      ]
  },
                      'FlipChange': {
      'text': 'When you divide a fraction by a whole number, the result should be a much smaller piece, not a larger one! You cannot treat the bottom number like a whole number. Instead, turn the whole number into a fraction (like 8/1) and use the "Keep-Change-Flip" rule to multiply by the reciprocal.',
      'links': [
          'https://www.youtube.com/watch?v=4lkq3DgvmGw', # Dividing fractions by whole numbers
          'https://www.youtube.com/watch?v=fpx7XkG0O3s'  # Keep Change Flip explained
      ]
  },
                      'Incorrect_equivalent_fraction_addition': {
      'text': 'You did the hard work of finding a common denominator—great job! However, remember that when you add fractions, the denominator stays the same. The bottom number tells you the "size" of the pieces, and adding two pieces of the same size doesn’t change the size of the slices, just how many you have!',
      'links': [
          'https://www.youtube.com/watch?v=bcCLlzS_6Yc', # Adding fractions with different denominators
          'https://www.youtube.com/watch?v=N-Y0KvgS6no'  # Why denominators stay the same
      ]
  },
                      'Wrong_fraction': {
      'text': 'Your math is perfect, but you found the number for the wrong group! The question asked for the "oatmeal" cookies, not the "chocolate chip" ones. After you find one group, remember to subtract it from the total, or start by using the leftover fraction (like 1/3 instead of 2/3).',
      'links': [
          'https://www.youtube.com/watch?v=7uV_G_Tf_YI', # Fraction word problems: Finding the remainder
          'https://www.youtube.com/watch?v=cbV83Xv6Kls'  # Complementary fractions
      ]
  },
                      'Wrong_Fraction': {
      'text': 'Your calculation is correct, but you found the number for the wrong group! The question asked for the "remainder" (like oatmeal or white chips), not the group mentioned in the fraction. You can fix this by subtracting your answer from the total, or by using the leftover fraction first.',
      'links': [
          'https://www.youtube.com/watch?v=7uV_G_Tf_YI', # Fraction of a set: The remainder
          'https://www.youtube.com/watch?v=cbV83Xv6Kls'  # Complementary fractions
      ]
  },
                      'Other':general_recommendation_map[problem_type],
                       'Irrelevant': general_recommendation_map[problem_type],
  'Wrong_Operation': {
    'text': 'It looks like you combined the numbers into a mixed fraction, but multiplication is different! To multiply a whole number by a fraction, you should multiply the whole number by the top (numerator) only. For example, 4 times 1/4 means you have four "quarter pieces," which equals 1 whole, not 4 and 1/4.',
    'links': [
        'https://www.youtube.com/watch?v=Rb4XW_V0Uvc', # Visualizing fraction x whole number
        'https://www.youtube.com/watch?v=17oG76m_36k'  # Why we don't just combine the numbers
    ]
},
                       'SwapDividend': {
    'text': 'The order of division is very important! You cannot swap the numbers just to make the division easier. When you divide a small fraction (like 1/3) by a whole number (like 21), your answer must be a very tiny piece, not a whole number like 7. Use the "Keep-Change-Flip" rule: Keep 1/3, Change to multiplication, and Flip 21 to 1/21.',
    'links': [
        'https://www.youtube.com/watch?v=vV38D_r7Y_E', # Understanding why order matters in division
        'https://www.youtube.com/watch?v=4lkq3DgvmGw'  # Dividing fractions by whole numbers
    ]
},
                       'Irrelevant_Correct': {
    'text': (
        "<strong>Atypical Explanation Detected:</strong> Your answer is correct! "
        "However, your explanation didn't quite match our expected patterns. "
        "Please review the general concept below to ensure your reasoning is solid "
        "and that you haven't relied on a coincidence to get the right answer."
    ),
    'links': general_recommendation_map.get(problem_type, {}).get('links', [])
}


  }
  return reccomendations_map[misconception]

def get_recommendations_educator(misconception, problem_type):
                       
    general_recommendation_map = {
        1: {
            'text': 'The student is confusing division by a fraction with regular division. They need reinforcement on the relationship between division and multiplication, specifically that dividing by a fraction is the same as multiplying by its reciprocal.',
            'links': [
                'https://www.youtube.com/watch?v=4lkq3DgvmGw', 
                'https://www.youtube.com/watch?v=fpx7XkG0O3s'
            ]
        },
        2: {
            'text': 'The student is not identifying the "common difference" correctly. They may be multiplying the first term instead of tracking the consistent growth between each step in the pattern.',
            'links': [
                'https://www.youtube.com/watch?v=V7m8m6U2u4M', 
                'https://www.youtube.com/watch?v=XpG-9T_7j0E'
            ]
        },
        3: {
            'text': 'The student does not understand that multiplying fractions means finding a "part of a part." Review that numerators multiply together and denominators multiply together—no common denominator is needed.',
            'links': [
                'https://www.youtube.com/watch?v=Rb4XW_V0Uvc', 
                'https://www.youtube.com/watch?v=qM2oW9_T7X0'
            ]
        },
        4: {
            'text': 'The student is not distinguishing between the amount mentioned and the "leftover" (remainder). Consider using bar models to help visualize which part of the whole they need to find.',
            'links': [
                'https://www.youtube.com/watch?v=7uV_G_Tf_YI', 
                'https://www.youtube.com/watch?v=cbV83Xv6Kls'
            ]
        },
        5: {
            'text': 'The student is attempting to add or subtract fractions without finding a common denominator. Emphasize that the denominators must represent pieces of the same size before combining.',
            'links': [
                'https://www.youtube.com/watch?v=bcCLlzS_6Yc', 
                'https://www.youtube.com/watch?v=N-Y0KvgS6no'
            ]
        },
        7: {
            'text': 'The student believes that longer decimal numbers are larger. They need to understand place value—that decimal value is determined by comparing columns (tenths, hundredths) from left to right, not by length.',
            'links': [
                'https://www.youtube.com/watch?v=cbV83Xv6Kls', 
                'https://www.youtube.com/watch?v=KzfWUEJjG18'
            ]
        },
        8: {
            'text': 'The student does not understand inverse proportion in work problems. They need to learn that more people means less time, and should calculate total "work" (People × Time) before dividing by the new number of workers.',
            'links': [
                'https://www.youtube.com/watch?v=H7mYVvXGisQ', 
                'https://www.youtube.com/watch?v=52Yf7_W_C-M'
            ]
        },
        9: {
            'text': 'The student is using subtraction when division is required. Review that division helps find how many equal parts a total can be split into.',
            'links': [
                'https://www.youtube.com/watch?v=vV38D_r7Y_E', 
                'https://www.youtube.com/watch?v=qM2oW9_T7X0'
            ]
        },
        10: {
            'text': 'The student does not understand the probability scale. Reinforce that all probabilities exist between 0 (Impossible) and 1 (Certain), and that decimals like 0.9 represent very high likelihood.',
            'links': [
                'https://www.youtube.com/watch?v=cbV83Xv6Kls', 
                'https://www.youtube.com/watch?v=KzfWUEJjG18'
            ]
        },
        11: {
            'text': 'The student is struggling with polygon angle relationships. Review that regular polygons follow specific angle rules: use the exterior angle (180 - interior) and the "360 rule" to calculate sides.',
            'links': [
                'https://www.youtube.com/watch?v=fS6uV_XmS9Y', 
                'https://www.youtube.com/watch?v=m7X3tK9S_Sg'
            ]
        },
        12: {
            'text': 'The student does not understand operations with negative numbers. Use a number line to demonstrate that subtracting a negative is the same as adding a positive.',
            'links': [
                'https://www.youtube.com/watch?v=qM2oW9_T7X0', 
                'https://www.youtube.com/watch?v=C38B33ZywWs'
            ]
        },
        13: {
            'text': 'The student is not separating variables and constants correctly. Review the concept of "like terms" and emphasize that solving for a variable requires performing inverse operations on both sides.',
            'links': [
                'https://www.youtube.com/watch?v=CLWpkv6S7_Y', 
                'https://www.youtube.com/watch?v=C_B_C8f5f5o'
            ]
        },
        14: {
            'text': 'The student does not understand equivalent fractions. Emphasize that both the numerator and denominator must be multiplied or divided by the same number—adding or subtracting changes the ratio.',
            'links': [
                'https://www.youtube.com/watch?v=qcS_S9_kC90', 
                'https://www.youtube.com/watch?v=N-Y0KvgS6no'
            ]
        },
        15: {
            'text': 'The student is confusing "part" and "whole" in fractions. They need to understand that the denominator represents all pieces in the shape, not just the unshaded ones.',
            'links': [
                'https://www.youtube.com/watch?v=n0FZhQ_GkKw', 
                'https://www.youtube.com/watch?v=3m693-hItp8'
            ]
        }
    }
    
    recommendations_map = {
        'Incomplete': {
            'text': 'The student is arriving at the correct answer but failing to simplify. Review when and how to reduce fractions to their simplest form.',
            'links': ['https://www.youtube.com/watch?v=aNQXhknSwrI']
        },
        'Longer_is_bigger': {
            'text': 'The student believes that more decimal places means a larger number. They need explicit instruction on place value comparison.',
            'links': ['https://www.youtube.com/watch?v=RHUl4kZDD6c', 'https://www.youtube.com/watch?v=trTS_KfkqtI']
        },
        'Shorter_is_bigger': {
            'text': 'The student believes that fewer decimal places means a larger number. Review place value and decimal comparison strategies.',
            'links': ['https://www.youtube.com/watch?v=RHUl4kZDD6c', 'https://www.youtube.com/watch?v=trTS_KfkqtI']
        },
        'Ignores_zeroes': {
            'text': 'The student is ignoring placeholder zeros between the decimal point and other digits (like the 0 in 8.07). Emphasize that these zeros affect place value.',
            'links': [
                'https://www.youtube.com/watch?v=trTS_KfkqtI',
                'https://www.youtube.com/watch?v=RHUl4kZDD6c'
            ]
        },
        'Whole_numbers_larger': {
            'text': 'The student assumes decimals are always smaller than whole numbers. Review that decimal comparisons require examining digits to the left of the decimal point first, and that the decimal part adds value.',
            'links': [
                'https://www.youtube.com/watch?v=0p_G69v4p9Y',
                'https://www.youtube.com/watch?v=RHUl4kZDD6c'
            ]
        },
        'Wrong_term': {
            'text': 'The student has identified the pattern rule but stopped calculating too early. They need to practice applying the rule repeatedly until reaching the requested term number.',
            'links': [
                'https://www.youtube.com/watch?v=XpG-9T_7j0E',
                'https://www.youtube.com/watch?v=V7m8m6U2u4M'
            ]
        },
        'Duplication': {
            'text': 'The student is multiplying both the numerator and denominator when multiplying a fraction by a whole number. Review that only the numerator should be multiplied.',
            'links': [
                'https://www.youtube.com/watch?v=Rb4XW_V0Uvc',
                'https://www.youtube.com/watch?v=17oG76m_36k'
            ]
        },
        'Division': {
            'text': 'The student is dividing when they should be multiplying to find a "fraction of a fraction." Teach that the word "of" signals multiplication in fraction problems.',
            'links': [
                'https://www.youtube.com/watch?v=qM2oW9_T7X0',
                'https://www.youtube.com/watch?v=vV38D_r7Y_E'
            ]
        },
        'Mult': {
            'text': 'The student multiplied instead of dividing. Review that dividing a fraction by a whole number creates smaller pieces, so the answer must be smaller than the starting fraction.',
            'links': [
                'https://www.youtube.com/watch?v=4lkq3Dgvm74',
                'https://www.youtube.com/watch?v=K3f6N-o4SAs'
            ]
        },
        'Certainty': {
            'text': 'The student does not understand probability terminology on the 0-1 scale. Review that only 1 is "Certain" and only 0 is "Impossible," with descriptive terms for values in between.',
            'links': [
                'https://www.youtube.com/watch?v=KzfWUEJjG18',
                'https://www.youtube.com/watch?v=AY3O_SGVtyE'
            ]
        },
        'Scale': {
            'text': 'The student misunderstands the probability scale magnitude. Emphasize that 0.75 or 0.95 are high probabilities because 1 is the maximum—anything over 0.5 is "Likely."',
            'links': [
                'https://www.youtube.com/watch?v=KzfWUEJjG18',
                'https://www.youtube.com/watch?v=cbV83Xv6Kls'
            ]
        },
        'Adding_terms': {
            'text': 'The student is adding a coefficient instead of recognizing multiplication. Review that when a number and letter are adjacent (7w), it means multiplication, requiring division to solve.',
            'links': [
                'https://www.youtube.com/watch?v=-_XG9H_mI3U',
                'https://www.youtube.com/watch?v=l3XzepN03KQ'
            ]
        },
        'Denominator-only_change': {
            'text': 'The student is changing only the denominator when finding common denominators. Emphasize that the numerator must be multiplied by the same factor to maintain equivalence.',
            'links': [
                'https://www.youtube.com/watch?v=bcCLlzS_6Yc',
                'https://www.youtube.com/watch?v=N-Y0KvgS6no'
            ]
        },
        'Inverse_operation': {
            'text': 'The student is not using inverse operations to solve equations. Review that solving requires "undoing" operations: division for multiplication, subtraction for addition, and vice versa.',
            'links': [
                'https://www.youtube.com/watch?v=Qyd_v3DGzTM',
                'https://www.youtube.com/watch?v=l3XzepN03KQ'
            ]
        },
        'Not_variable': {
            'text': 'The student does not understand that variables represent unknown values. They may be treating the letter as a placeholder for a digit rather than a complete number requiring algebraic solving.',
            'links': [
                'https://www.youtube.com/watch?v=vDqOoVux9Zg',
                'https://www.youtube.com/watch?v=l3XzepN03KQ'
            ]
        },
        'Adding_across': {
            'text': 'The student is adding denominators together. Use the pizza analogy: denominators show slice size, which must be standardized before combining. Only numerators are added.',
            'links': [
                'https://www.youtube.com/watch?v=bcCLlzS_6Yc',
                'https://www.youtube.com/watch?v=N-Y0KvgS6no'
            ]
        },
        'Subtraction': {
            'text': 'The student is subtracting when multiplication is needed for "fraction of a fraction" problems. Review that "of" signals multiplication, even in contexts involving sharing or removing.',
            'links': [
                'https://www.youtube.com/watch?v=qM2oW9_T7X0',
                'https://www.youtube.com/watch?v=vV38D_r7Y_E'
            ]
        },
        'Firstterm': {
            'text': 'The student is incorrectly multiplying the first term by the position number. Review that arithmetic sequences grow by repeated addition of the common difference, not multiplication.',
            'links': [
                'https://www.youtube.com/watch?v=XpG-9T_7j0E',
                'https://www.youtube.com/watch?v=V7m8m6U2u4M'
            ]
        },
        'Interior': {
            'text': 'The student is struggling to use interior angles correctly. Teach the strategy of finding the exterior angle first (180 - interior), then using the 360° rule for regular polygons.',
            'links': [
                'https://www.youtube.com/watch?v=fS6uV_XmS9Y',
                'https://www.youtube.com/watch?v=m7X3tK9S_Sg'
            ]
        },
        'Unknowable': {
            'text': 'The student does not understand that in regular polygons, all interior angles are equal. Review that one angle provides sufficient information to determine the number of sides.',
            'links': [
                'https://www.youtube.com/watch?v=fS6uV_XmS9Y',
                'https://www.youtube.com/watch?v=m7X3tK9S_Sg'
            ]
        },
        'Inversion': {
            'text': 'The student is multiplying the denominator when they should only multiply the numerator. Review that multiplying the bottom divides the fraction, making it smaller.',
            'links': [
                'https://www.youtube.com/watch?v=Rb4XW_V0Uvc',
                'https://www.youtube.com/watch?v=17oG76m_36k'
            ]
        },
        'Positive': {
            'text': 'The student is converting both numbers to positive when subtracting negatives. Review that only the "minus a negative" becomes plus—the first number retains its original sign.',
            'links': [
                'https://www.youtube.com/watch?v=qM2oW9_T7X0',
                'https://www.youtube.com/watch?v=C38B33ZywWs'
            ]
        },
        'Additive': {
            'text': 'The student is adding or subtracting to create equivalent fractions instead of multiplying or dividing. Emphasize that equivalent fractions require proportional scaling of both parts.',
            'links': [
                'https://www.youtube.com/watch?v=N-Y0KvgS6no',
                'https://www.youtube.com/watch?v=qcS_S9_kC90'
            ]
        },
        'Base_rate': {
            'text': 'The student does not understand inverse proportion in work-rate problems. Teach the strategy of finding total work (people × time), then dividing by the new workforce.',
            'links': [
                'https://www.youtube.com/watch?v=H7mYVvXGisQ',
                'https://www.youtube.com/watch?v=52Yf7_W_C-M'
            ]
        },
        'WNB': {
            'text': 'The student is using the wrong numbers for the numerator and denominator. Review that fractions show "part over whole"—the denominator must represent all pieces, not just unshaded ones.',
            'links': [
                'https://www.youtube.com/watch?v=n0FZhQ_GkKw',
                'https://www.youtube.com/watch?v=3m693-hItp8'
            ]
        },
        'Multiplying_by_4': {
            'text': 'The student is incorrectly scaling in work problems. Review the inverse relationship: more workers means less time. Calculate total work first, then divide.',
            'links': [
                'https://www.youtube.com/watch?v=H7mYVvXGisQ',
                'https://www.youtube.com/watch?v=52Yf7_W_C-M'
            ]
        },
        'Tacking': {
            'text': 'The student is treating negative signs as labels rather than directional operators. Use the number line and "Keep-Change-Change" rule to demonstrate proper integer operations.',
            'links': [
                'https://www.youtube.com/watch?v=C38B33ZywWs',
                'https://www.youtube.com/watch?v=qM2oW9_T7X0'
            ]
        },
        'Definition': {
            'text': 'The student assumes "polygon" has a fixed number of sides. Review that polygons are defined by having straight sides, and the specific angle information determines the side count.',
            'links': [
                'https://www.youtube.com/watch?v=fS6uV_XmS9Y',
                'https://www.youtube.com/watch?v=m7X3tK9S_Sg'
            ]
        },
        'FlipChange': {
            'text': 'The student is not applying the "Keep-Change-Flip" rule correctly for dividing fractions by whole numbers. The result should be much smaller than the starting fraction.',
            'links': [
                'https://www.youtube.com/watch?v=4lkq3DgvmGw',
                'https://www.youtube.com/watch?v=fpx7XkG0O3s'
            ]
        },
        'Incorrect_equivalent_fraction_addition': {
            'text': 'The student found a common denominator but is incorrectly adding both numerator and denominator. Review that the denominator represents piece size and stays constant during addition.',
            'links': [
                'https://www.youtube.com/watch?v=bcCLlzS_6Yc',
                'https://www.youtube.com/watch?v=N-Y0KvgS6no'
            ]
        },
        'Wrong_fraction': {
            'text': 'The student performed the calculation correctly but identified the wrong group. Review complementary fractions and how to find remainders by subtraction or using the leftover fraction.',
            'links': [
                'https://www.youtube.com/watch?v=7uV_G_Tf_YI',
                'https://www.youtube.com/watch?v=cbV83Xv6Kls'
            ]
        },
        'Wrong_Fraction': {
            'text': 'The student calculated accurately but answered for the mentioned group instead of the remainder. Practice identifying which quantity the question is actually requesting.',
            'links': [
                'https://www.youtube.com/watch?v=7uV_G_Tf_YI',
                'https://www.youtube.com/watch?v=cbV83Xv6Kls'
            ]
        },
        'Other': general_recommendation_map[problem_type],
        'Irrelevant': general_recommendation_map[problem_type],
        'Wrong_Operation': {
            'text': 'The student is combining numbers into a mixed fraction instead of multiplying. Review that fraction multiplication means multiplying the numerator only by the whole number.',
            'links': [
                'https://www.youtube.com/watch?v=Rb4XW_V0Uvc',
                'https://www.youtube.com/watch?v=17oG76m_36k'
            ]
        },
        'SwapDividend': {
            'text': 'The student is reversing the order of division to make it easier. Emphasize that order matters: dividing a small fraction by a large number must yield a tiny result, not a whole number.',
            'links': [
                'https://www.youtube.com/watch?v=vV38D_r7Y_E',
                'https://www.youtube.com/watch?v=4lkq3DgvmGw'
            ]
        },
        'Irrelevant_Correct': {
            'text': (
                "<strong>Atypical Explanation Detected:</strong> Your answer is correct! "
                "However, your explanation didn't quite match our expected patterns. "
                "Please review the general concept below to ensure your reasoning is solid."
            ),
            'links': [] # Keep this as an empty placeholder
        }
    } # Closes recommendations_map

    
    return recommendations_map.get(misconception, general_recommendation_map.get(problem_type, {
        'text': 'The student is struggling with fundamental concepts for this problem type. Additional review is recommended.',
        'links': []
    }))

def present_mc_answers(grouped):
    """
    Creates a dictionary mapping question texts to their multiple choice options.

    Args:
        grouped: DataFrame with 'QuestionText' and 'context_values' columns

    Returns:
        Dictionary where keys are question texts and values are dicts mapping letters to answers
        Example: {"Question text": {"A": "answer1", "B": "answer2", ...}}
    """
    choices_dict = {}

    for ind, row in grouped.iterrows():
        question_text = row['QuestionText']
        context_values = row['context_values']

        # Create letter mappings (A, B, C, D, etc.)
        letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J','K','L','M','N','O']

        # Map each value to a letter
        letter_dict = {}
        for i, value in enumerate(context_values):
            if i < len(letters):
                letter_dict[letters[i]] = value

        choices_dict[question_text] = letter_dict

    return choices_dict
def generate_final_assessment(grouped,testing=False):
    count = 1
    choices_dict = present_mc_answers(grouped)

    for ind, row in grouped.iterrows():
        prob_type = row['problem_type']
        question_text = row['QuestionText']
        choices = choices_dict[question_text]
        all_possible_answers = list(choices.keys())

        # Format choices string
        choice_str = ""
        for i, (letter, choice) in enumerate(choices.items()):
            choice_str += f"\t{letter}. {choice}"
            if i < len(choices) - 1:
                choice_str += "\n"

        print()
        print("-" * 30)
        print(f"Question {count}: {row['QuestionText']}")
        print()
        print(f"     Possible Answers: \n{choice_str}")

        # Keep asking until valid answer is received
        valid_answer = False
        while not valid_answer:
            try:
                units=extract_unit(row['QuestionText'])

                if prob_type == 8:
                    user_val = input(f"     Answer ({units}): ")

                else:
                    user_val = input("     Answer: ")

                if user_val not in all_possible_answers:
                    raise ValueError

                # If we get here, the answer is valid
                valid_answer = True

            except (ValueError, KeyError):
                joined_answers = ", ".join(all_possible_answers)
                print(f'\n     ❌ Invalid input. Please enter one of: {joined_answers}')
                print(f'     Please try again.\n')

        # Convert letter choice to actual answer text
        user_val = choices_dict[question_text][user_val]
        user_explanation = input("     Explanation: ")
        if testing:
          misc=input('Misconception: ')
          category=input('Category: ')
          grouped.at[ind, 'Misconception'] = [c for c in misc.split(',')]
          grouped.at[ind, 'Category'] = [category]
        count += 1
        grouped.at[ind, 'MC_Answer'] = user_val
        grouped.at[ind, 'StudentExplanation'] = user_explanation

        print("-" * 30)

    return grouped
def apply_feature_substitution(row):
  explanation=row['StudentExplanation']
  math_context_string=row['Mathematical_Context']
  problem_type=row['problem_type']
  feature_map, value_map = augment_functions.parse_and_map_features(math_context_string)
  new_explanation = augment_functions.place_generic_features_in_text(value_map, explanation,problem_type)
  return new_explanation


# ============================================================================
# DUAL FEATURE DATASET CLASS
# ============================================================================

class DualFeatureDataset(torch.utils.data.Dataset):
    """Dataset that returns both full and reduced mathematical context"""

    def __init__(self, df, tokenizer, label_encoder_category, label_encoder_misconception,
                 augmenter, max_length=450, kind='val'):
        self.kind = kind
        self.label_encoder_category = label_encoder_category
        self.label_encoder_misconception = label_encoder_misconception
        self.questions = df['QuestionText'].tolist()
        self.explanations = df['StudentExplanation'].tolist()
        self.correctness = df['Correctness'].tolist()
        self.mc_answer = df['MC_Answer'].tolist()
        self.categories = df['Category'].tolist()
        self.misconceptions = df['Misconception'].tolist()
        self.math_context_full = df['Mathematical_Context'].tolist()
        self.math_context_reduced = df['Mathematical_Context_Reduced'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augmenter = augmenter

    def __len__(self):
        return len(self.questions)

    def format_math_context_as_text(self, context_string):
        """Convert math context to text format"""
        if not context_string:
            return ""

        parts = [part.strip() for part in context_string.split('|') if part.strip()]
        formatted_features = []

        for part in parts:
            if ':' in part:
                key, value = part.split(':', 1)
                formatted_features.append(f"{key.strip()}:{value.strip()}")

        if not formatted_features:
            return ""

        return " <MATH_CONTEXT> " + " | ".join(formatted_features) + " </MATH_CONTEXT> "

    def __getitem__(self, idx):
        question = self.questions[idx]
        explanation = self.explanations[idx]
        correct_label = self.correctness[idx]
        category_labels = self.categories[idx]
        mc_answer = str(self.mc_answer[idx])
        misconception_label = self.misconceptions[idx]

        # Get both full and reduced contexts
        math_context_full = self.format_math_context_as_text(self.math_context_full[idx])
        math_context_reduced = self.format_math_context_as_text(self.math_context_reduced[idx])

        question_with_context = question + " " + mc_answer

        # Encode category label
        if isinstance(category_labels, list):
            category_labels = category_labels[0]
        category_label = self.label_encoder_category.transform([category_labels])[0]

        # Encode misconception labels
        misconception_multi_label = [0] * len(self.label_encoder_misconception.classes_)
        if isinstance(misconception_label, str):
            misconception_label = [misconception_label]
        for mis in misconception_label:
            if mis in self.label_encoder_misconception.classes_:
                idx_mis = self.label_encoder_misconception.transform([mis])[0]
                misconception_multi_label[idx_mis] = 1

        # Create inputs with FULL context (for correctness model)
        explanation_full = math_context_full + " " + explanation
        correctness_input1=question+" "+math_context_full
        correctness_input2=mc_answer
        encoding_full = self.tokenizer(
            correctness_input1,
            correctness_input2,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Create inputs with REDUCED context (for misconception model)
        explanation_reduced = math_context_reduced + " " + explanation
        encoding_reduced = self.tokenizer(
            question_with_context,
            explanation_reduced,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids_full': encoding_full['input_ids'].squeeze(0),
            'attention_mask_full': encoding_full['attention_mask'].squeeze(0),
            'input_ids_reduced': encoding_reduced['input_ids'].squeeze(0),
            'attention_mask_reduced': encoding_reduced['attention_mask'].squeeze(0),
            'correctness': torch.tensor(correct_label, dtype=torch.long),
            'category': torch.tensor(category_label, dtype=torch.long),
            'misconception': torch.tensor(misconception_multi_label, dtype=torch.float32),
            'question_text': question_with_context,
            'explanation_text': explanation
        }


def filter_math_context_case_insensitive(context_string):
    if not isinstance(context_string, str):
        return ""

    # Regex breakdown:
    # \w+          -> The feature name (letters/numbers/underscores)
    # \s*:\s* -> The colon, allowing for optional spaces around it
    # (?:TRUE|FALSE) -> The literal words TRUE or FALSE
    # flags=re.I   -> Makes the whole search ignore case (T/t, R/r, etc.)

    pattern = r'\w+\s*:\s*(?:TRUE|FALSE)'
    boolean_features = re.findall(pattern, context_string, flags=re.I)

    # Re-join with pipes for a clean, scannable string
    return " | ".join(boolean_features)
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
def extract_unit(text: str) -> str:
    # Expanded list based on your specific problem types
    units_list = [
        'days', 'hours', 'minutes', 'seconds', 'weeks'

    ]

    text_lower = str(text).lower()

    # Strategy: Find any unit from the list that appears in the text.
    # We use \b to ensure we don't match 'day' inside 'yesterday'.
    pattern = r'\b(' + '|'.join(units_list) + r')\b'

    matches = re.findall(pattern, text_lower)

    if matches:
        # If there are multiple units (like 'workers' and 'days'),
        # we usually want the one associated with the quantity being asked for.
        # Often, the LAST mentioned unit in the setup is the one being solved for.
        return matches[0]

    return ""

def get_testing_sample(augmented_data_set):
  return augmented_data_set.groupby('problem_type').agg({col:'first' for col in augmented_data_set.columns if col!='problem_type'}).reset_index(drop=False)
def run_assessment_app(report_for='educator'):
    DRIVE_DIR = '/content/drive/MyDrive/'

    # 3. Create the directory if it doesn't exist (good practice for saving checkpoints)
    if not os.path.exists(DRIVE_DIR):
        os.makedirs(DRIVE_DIR)
        print(f"Created directory: {DRIVE_DIR}")
    else:
        print(f"Using existing directory: {DRIVE_DIR}")
    drive_dir=DRIVE_DIR
    project_dir=os.path.join(drive_dir,'LLM_Misconception_Project')
    checkpoint_dir=os.path.join(project_dir,'Checkpoints')

    augmented_data_set = augment_functions.generate_all_augmented_data(change_cat_prob=0.0, sample_size=30)
    augmented_data_set['Original_category'] = augmented_data_set['Category'].apply(lambda x: x[0])

    label_encoder_category = joblib.load(os.path.join(scripts_dir, 'label_encoder_category.joblib'))
    label_encoder_misconception = joblib.load(os.path.join(scripts_dir, 'label_encoder_misconception.joblib'))

    #grouped = generate_assessment_questions(augmented_data_set)
    C=augmented_data_set[augmented_data_set['problem_type'].isin([1,3,5])]
    C=C[~C['QuestionText'].str.contains('1')]
    grouped=C.sample(15)
    grouped['MC_Answer'] = 'None'
    grouped['Mathematical_Context'] = grouped[['QuestionText','MC_Answer','problem_type']].apply(
        lambda x: augment_functions.create_generic_math_context(x['QuestionText'], x['MC_Answer'], x['problem_type']),
        axis=1
    )
    grouped['context_values'] = grouped.apply(lambda x: extract_misconception_values(x), axis=1)

    my_device = torch.device('cuda')

    # Use the new shared-base function
    grouped_with_predictions = assessment_inference_shared_base(  # Changed function name
        grouped=grouped,
        kind='inference',
        correctness_checkpoint_path=os.path.join(checkpoint_dir, 'stage2_explanation_model_dec5.pt'),
        misconception_checkpoint_path=os.path.join(checkpoint_dir, 'stage2_explanation_model_dec5_MISC_ONLY_SUB.pt'),
        model_name='Qwen/Qwen2-7B-Instruct',
        label_encoder_category=label_encoder_category,
        label_encoder_misconception=label_encoder_misconception,
        device=my_device,
        batch_size=16,
        add_ground_truth=True
    )

    display_final_results(grouped_with_predictions,report_for=report_for)
    return grouped_with_predictions


"""
Gradio App for Math Misconception Detection
File: app.py

This replaces the terminal-based input system with a web interface.
Place this in your Hugging Face Space repository.
"""

# ============================================================================
# GLOBAL STATE MANAGEMENT
# ============================================================================

class AssessmentState:
    """Manages the state of the assessment across multiple interactions"""
    def __init__(self):
        self.grouped_df = None
        self.current_question_idx = 0
        self.choices_dict = None
        self.total_questions = 0
        self.completed = False
        self.results_df=None
        
        
    def reset(self):
        self.grouped_df = None
        self.current_question_idx = 0
        self.choices_dict = None
        self.total_questions = 0
        self.completed = False
        self.results_df=None  # ADD THIS

# Global state
state = AssessmentState()

# ============================================================================
# MODEL LOADING (happens once at startup)
# ============================================================================

def load_models_and_data():
    """Load label encoders only - models loaded on-demand"""
    print("🔄 Loading label encoders...")
    
    label_encoder_category = joblib.load(
        os.path.join(models_dir, 'label_encoder_category.joblib')
    )
    label_encoder_misconception = joblib.load(
        os.path.join(models_dir, 'label_encoder_misconception.joblib')
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # REMOVED THESE TWO LINES:
    # correctness_checkpoint_path = get_model_path("stage2_explanation_model_dec5.pt")
    # misconception_checkpoint_path = get_model_path("stage2_explanation_model_dec5_MISC_ONLY_SUB.pt")
    
    return {
        'label_encoder_category': label_encoder_category,
        'label_encoder_misconception': label_encoder_misconception,
        'device': device
        # REMOVED: correctness_checkpoint_path
        # REMOVED: misconception_checkpoint_path
    }  
  

# Load models once at startup
MODELS = load_models_and_data()

# ============================================================================
# GRADIO INTERFACE FUNCTIONS
# ============================================================================

def get_model_path(filename: str, repo_id: str = "jprich1984/math-misconception-models") -> str:
    """
    Download model from HuggingFace Models repo if needed.
    Falls back to local path for Colab compatibility.
    
    Args:
        filename: Name of the checkpoint file
        repo_id: HuggingFace repo ID
    
    Returns:
        Local path to the model file
    """
    # Check if running on HF Spaces
    if 'SPACE_ID' in os.environ:
        print(f"📥 Downloading {filename} from HuggingFace...")
        try:
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir="./model_cache"
            )
            print(f"✅ Downloaded to: {model_path}")
            return model_path
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")
            raise
    else:
        # Running in Colab - use ABSOLUTE path
        # First, determine the project root
        if 'LLM_Misconception_Project' in os.getcwd():
            # Already inside project directory
            project_root = os.getcwd().split('LLM_Misconception_Project')[0] + 'LLM_Misconception_Project'
        else:
            # Assume standard Colab location
            project_root = '/content/LLM_Misconception_Project'
        
        local_path = os.path.join(project_root, "checkpoints", filename)
        
        if os.path.exists(local_path):
            print(f"✅ Using local model: {local_path}")
            return local_path
        else:
            # File doesn't exist locally - try downloading
            print(f"⚠️ Local file not found at: {local_path}")
            print(f"📥 Attempting to download {filename} from HuggingFace...")
            return hf_hub_download(repo_id=repo_id, filename=filename)

def filter_polygon_questions(df):
    """
    Filter out rows where problem_type is 11 and QuestionText doesn't contain 'regular'.
    
    Args:
        df: DataFrame with 'problem_type' and 'QuestionText' columns
    
    Returns:
        Filtered DataFrame
    """
    # Keep all rows except problem_type 11 without 'regular'
    mask = ~((df['problem_type'] == 11) & (~df['QuestionText'].str.contains('regular', case=False, na=False)))
    return df[mask]
def get_cleaned_dataframe(df):
    """
    Returns the full dataframe minus the specific rows where 
    problem_type is 7 AND the QuestionText lacks comparison keywords.
    """
    words = ['bigger', 'biggest', 'largest', 'larger', 'greatest', 'greater']
    
    # 1. Identify rows that are problem_type 7
    is_type_7 = df['problem_type'] == 7
    
    # 2. Identify rows that LACK the keywords (case-insensitive)
    # We use a regex pattern for speed and 'case=False' for safety
    pattern = '|'.join(words)
    lacks_keywords = ~df['QuestionText'].str.contains(pattern, case=False, na=False)
    
    # 3. Create the 'Bad Rows' mask: Type 7 AND No Keywords
    bad_rows_mask = is_type_7 & lacks_keywords
    
    # 4. Return the dataframe excluding the bad rows
    return df[~bad_rows_mask].copy()

def get_cleaned_dataframe_v2(df):
    """
    Returns the full dataframe minus rows where:
    problem_type is 1 AND the QuestionText does not contain '1'.
    """
    # 1. Identify rows that are problem_type 1
    is_type_1 = df['problem_type'] == 1
    
    # 2. Identify rows that DO NOT contain the character '1'
    # (na=False ensures we don't drop rows with empty/NaN text)
    does_not_have_1 = ~df['QuestionText'].str.contains('1', na=False)
    
    # 3. Target only the intersection: Type 1 AND No '1'
    rows_to_drop = is_type_1 & does_not_have_1
    
    # 4. Return the dataframe excluding those specific rows
    return df[~rows_to_drop].copy()
def start_assessment(num_questions: int = 14, progress=gr.Progress()) -> Tuple[str, gr.update, gr.update, gr.update, gr.update]:
    """
    Initialize a new assessment session with progress tracking.
    Returns: (question_html, answer_radio, explanation_box, submit_button, start_button)

    """

    try:
        progress(0, desc="⏳ Initializing assessment")
        state.reset()
        
        progress(0.2, desc="📝 Generating questions...")
        augmented_data_set = augment_functions.generate_all_augmented_data(
            change_cat_prob=0.0, 
            sample_size=30,
            which='val'
        )
        augmented_data_set=filter_polygon_questions(augmented_data_set)
        augmented_data_set=get_cleaned_dataframe(augmented_data_set)
        augmented_data_set=get_cleaned_dataframe_v2(augmented_data_set)
        progress(0.5, desc="🎯 Selecting questions...")
        grouped=augmented_data_set.groupby('problem_type').sample(1)
      
        
        progress(0.7, desc="✏️ Preparing answer choices")
        grouped['MC_Answer'] = 'None'
        grouped['Mathematical_Context'] = grouped[['QuestionText', 'MC_Answer', 'problem_type']].apply(
            lambda x: augment_functions.create_generic_math_context(
                x['QuestionText'], x['MC_Answer'], x['problem_type']
            ),
            axis=1
        )
        
        progress(0.9, desc="🔧 Finalizing...")
        grouped['context_values'] = grouped.apply(lambda x: extract_misconception_values(x), axis=1)
        grouped['StudentExplanation'] = ''
        
        state.grouped_df = grouped.reset_index(drop=True)
        state.choices_dict = present_mc_answers(grouped)
        state.total_questions = len(grouped)
        state.current_question_idx = 0
        
        progress(1.0, desc="✅ Ready!")
        
        # Get first question
        question_html, choices = get_current_question()
        
        return (
            question_html,
            gr.update(choices=choices, value=None, visible=True),
            gr.update(value="", visible=True),
            gr.update(visible=True),
            gr.update(visible=False)  # Hide start button
        )
        
    except Exception as e:
        error_html = f"""
        <div style="padding: 20px; background: #f8d7da; border-radius: 10px; color: #721c24;">
            <h3>❌ Error starting assessment</h3>
            <p>{str(e)}</p>
        </div>
        """
        return (
            error_html,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True)  # Show start button again
        )



def get_current_question() -> Tuple[str, List[str]]:
    """Get the current question and its choices"""
    if state.grouped_df is None or state.current_question_idx >= state.total_questions:
        return "<h2>No active assessment</h2>", []
    
    row = state.grouped_df.iloc[state.current_question_idx]
    
    # Keep original for dictionary lookup
    original_question = row['QuestionText']
    
    # Convert for display
    question_display = render_latex_to_html(original_question)
    
    # Use ORIGINAL text to get choices
    choices = state.choices_dict[original_question]
    
    question_html = f"""
    <div style="padding: 20px; background: #f8f9fa; border-radius: 10px; margin-bottom: 20px;">
        <h2>Question {state.current_question_idx + 1} of {state.total_questions}</h2>
        <div style="font-size: 18px; margin: 20px 0;">
            {question_display}
        </div>
    </div>
    """
    
    choice_list = [f"{letter}: {value}" for letter, value in choices.items()]
    
    return question_html, choice_list

def submit_answer(selected_answer: str, explanation: str, request: gr.Request, progress=gr.Progress()) -> Tuple[str, str, gr.update, gr.update, gr.update, gr.update]:
    """
    Process the submitted answer and move to next question.
    Returns: (question_html, feedback_html, answer_radio, explanation_box, submit_button, show_assessment_btn)
    """
    if state.grouped_df is None:
        return (
            "<h2>Please start an assessment first.</h2>",
            "",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(visible=False)
        )
    
    if not selected_answer or not explanation:
        return (
            gr.update(),
            "⚠️ Please select an answer and provide an explanation.",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(visible=False)
        )
    
    # Extract the letter from "A: answer text"
    answer_letter = selected_answer.split(':')[0].strip()
    
    # Get the actual answer value
    current_row = state.grouped_df.iloc[state.current_question_idx]
    question_text = current_row['QuestionText']
    answer_value = state.choices_dict[question_text][answer_letter]
    
    # Store the answer
    state.grouped_df.at[state.current_question_idx, 'MC_Answer'] = answer_value
    state.grouped_df.at[state.current_question_idx, 'StudentExplanation'] = explanation
    
    # Move to next question
    state.current_question_idx += 1
    
    if state.current_question_idx >= state.total_questions:
        state.completed = True
        
        results_html, results_df= finish_assessment(request=request, progress=progress)
        state.results_df = results_df
        return (
            "<h2>✅ Assessment Complete!</h2>",
            results_html,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True)  # Show assessment button
        )
    
    # Get next question
    question_html, choices = get_current_question()
    feedback = f"""
    <div style="padding: 15px; background: #d4edda; border-radius: 5px; color: #155724;">
        ✅ Answer recorded! Moving to question {state.current_question_idx + 1}...
    </div>
    """
    
    return (
        question_html,
        feedback,
        gr.update(choices=choices, value=None),
        gr.update(value=""),
        gr.update(),
        gr.update(visible=False)  # Keep button hidden during assessment
    )
# ============================================================================
# GPU DECORATOR (Applied just before use)
# ============================================================================
if IS_ON_SPACES and SPACES_AVAILABLE:
    gpu_decorator = spaces.GPU(duration=120)
    print("✅ GPU decorator enabled")
else:
    gpu_decorator = lambda x: x
    print("ℹ️ GPU decorator disabled")

# Now define the decorated function immediately after
@gpu_decorator
def finish_assessment(request: gr.Request, progress=gr.Progress()) -> str:  # ADD request parameter
    """
    Run the AI analysis on all answers with progress tracking.
    Returns: results_html
    """
    global MODELS
    global results_cache 
    try:
        progress(0, desc="Loading AI models...")
        
        # LOAD MODELS HERE (moved from load_models_and_data)
        correctness_checkpoint_path = get_model_path('stage2_explanation_model_dec5_restructure.pt')
        misconception_checkpoint_path = get_model_path("stage2_explanation_model_dec5_MISC_ONLY_SUB_Other.pt")
        
        
        progress(0.2, desc="Preprocessing student responses...")
        
        progress(0.4, desc="Running AI analysis...")
        results_df = assessment_inference_shared_base(
            grouped=state.grouped_df,
            kind='inference',
            correctness_checkpoint_path=correctness_checkpoint_path ,
            misconception_checkpoint_path=misconception_checkpoint_path,
            model_name='Qwen/Qwen2-7B-Instruct',
            label_encoder_category=MODELS['label_encoder_category'],
            label_encoder_misconception=MODELS['label_encoder_misconception'],
            device=MODELS['device'],
            batch_size=16,
            add_ground_truth=False
        )
        
        # Save results
        if 'SPACE_ID' not in os.environ:
            file_suffix = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path = os.path.join(project_root, 'data', f"Inference_Results_{file_suffix}.csv")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            results_df.to_csv(save_path)
            print(f"💾 Results saved to: {save_path}")
        progress(0.9, desc="Generating recommendations...")
        results_html = format_results_as_html(results_df)
        
        results_cache['results_df'] = results_df  # ← cache only
        
        progress(1.0, desc="Complete!")
        return results_html, results_df
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"""
        <div style="padding: 20px; background: #f8d7da; border-radius: 10px; color: #721c24;">
            <h2>❌ Error During Analysis</h2>
            <p><strong>Error:</strong> {str(e)}</p>
            <details>
                <summary>Technical Details (click to expand)</summary>
                <pre style="background: white; padding: 10px; overflow: auto;">{error_details}</pre>
            </details>
        </div>
        """, None


def render_feedback_item(label, res):
    """Helper to render individual feedback blocks inside a question card."""
    clean_label = str(label).replace('_', ' ')
    links_html = ""
    if res.get('links'):
        links_list = "".join([f'<li><a href="{l}" target="_blank" style="color: #3182ce; text-decoration: none;">▶️ Video Tutorial {i}</a></li>' for i, l in enumerate(res['links'], 1)])
        links_html = f'<ul style="list-style: none; padding: 0; margin-top: 10px;">{links_list}</ul>'
    
    return f"""
    <div style="margin-top: 15px; padding: 15px; background: rgba(255,255,255,0.6); border-radius: 8px;">
        <h4 style="margin: 0 0 10px 0; color: #2c3e50; text-transform: uppercase; font-size: 13px; letter-spacing: 1px;">🚨 {clean_label}</h4>
        <p style="margin: 0; line-height: 1.6;">{res.get('text', 'No specific details available.')}</p>
        {links_html}
    </div>
    """

def format_results_as_html(results_df: pd.DataFrame, report_for: str = 'student') -> str:
    # 1. Summary Header
    total_questions = len(results_df)
    is_correct_mask = results_df['predicted_correctness'].apply(lambda x: str(x).strip().lower() in ['correct', '1'])
    correct_count = is_correct_mask.sum()
    percent_correct = (correct_count / total_questions) * 100 if total_questions > 0 else 0
    
    html = f"""
    <div style="padding: 20px; font-family: 'Segoe UI', sans-serif; max-width: 900px; margin: auto; color: #333;">
        <h1 style="color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px;">📊 Personalised Math Feedback</h1>
        <div style="background: #f1f9ff; padding: 20px; border-radius: 12px; margin-bottom: 30px; border: 1px solid #d1e9ff;">
            <p style="font-size: 18px; margin: 5px 0;"><strong>Score:</strong> {correct_count} / {total_questions}</p>
            <p style="font-size: 24px; margin: 5px 0; color: #1f618d;"><strong>Accuracy: {percent_correct:.1f}%</strong></p>
        </div>
    """
    
    vague_labels = {'Other', 'Irrelevant','Multiplying_by_4'}
    ignore_labels = {'No Misconception','none', 'None', None}
    any_feedback_given = False

    for ind, row in results_df.iterrows():
        problem_type = row['problem_type']
        is_correct = str(row['predicted_correctness']).strip().lower() in ['correct', '1']
        status_text = "✅ CORRECT" if is_correct else "❌ INCORRECT"
        
        m_list = row['predicted_misconception_all']
        if not isinstance(m_list, list): m_list = [m_list]

        specific_ones = [m for m in m_list if m not in vague_labels and m not in ignore_labels]
        has_vague = any(m in vague_labels for m in m_list)
        
        # Determine the display queue: (Header, Lookup_Key)
        display_queue = []

        # Case A: Specific Misconceptions found
        for m in specific_ones:
            display_queue.append((f"{status_text} | Misconception Detected: {m}", m))

        # Case B: Correct Answer, but Vague/Atypical Explanation (The "Irrelevant_Correct" path)
        if is_correct and has_vague and not specific_ones:
            display_queue.append((f"{status_text} | Atypical Reasoning Detected", "Irrelevant_Correct"))

        # Case C: Incorrect Answer, Vague/Other
        elif not is_correct and (has_vague or not specific_ones):
            display_queue.append((f"{status_text} | General Topic Review", "Other"))

        if not display_queue:
            continue
            
        any_feedback_given = True
        bg = "#fffdf0" if is_correct else "#fff5f5"
        border = "#fef3c7" if is_correct else "#feb2b2"

        # Render LaTeX in question text
        question_display = render_latex_to_html(row['QuestionText'])

        html += f"""
        <div style="margin: 40px 0; padding: 25px; background: {bg}; border: 1px solid {border}; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
            <div style="margin-bottom: 20px; padding-bottom: 15px; border-bottom: 1px dashed {border};">
                <p><strong>❓ Question:</strong> {question_display}</p>
                <p><strong>📝 Your Answer:</strong> {row.get('MC_Answer', 'N/A')}</p>
                <p><strong>💭 Your Explanation:</strong> <em>"{row.get('StudentExplanation_Original', 'N/A')}"</em></p>
            </div>
        """

        for display_header, lookup_key in display_queue:
            # Fetch recommendations
            if lookup_key == "Irrelevant_Correct":
                # Get the "Other" (General) info but use your custom text for the correct/atypical case
                gen_res = get_recommendations_educator('Other', problem_type) if report_for == 'educator' else get_recommendations('Other', problem_type)
                res = {
                    'text': f"<strong>Note:</strong> Your answer is correct, but your explanation was flagged as atypical. Please review this general concept to ensure your understanding is complete.<br><br><strong>General Review:</strong> {gen_res.get('text')}",
                    'links': gen_res.get('links', [])
                }
            else:
                res = get_recommendations_educator(lookup_key, problem_type) if report_for == 'educator' else get_recommendations(lookup_key, problem_type)
            
            html += render_feedback_item(display_header, res)

        html += "</div>"

    if not any_feedback_given:
        html += '<div style="background: #f0fff4; padding: 30px; text-align: center; border-radius: 15px;">🎉 <strong>Amazing!</strong> No misconceptions found.</div>'

    html += "</div>"
    return html
# ============================================================================
# GRADIO INTERFACE
# ============================================================================
from latex2mathml.converter import convert

def render_latex_to_html(text):
    """Convert LaTeX notation to rendered HTML"""
    import re
    
    # Find all \(...\) patterns
    def replace_inline(match):
        latex = match.group(1)
        try:
            mathml = convert(latex)
            return mathml
        except:
            return match.group(0)  # Return original if conversion fails
    
    # Replace inline math \(...\)
    text = re.sub(r'\\\((.*?)\\\)', replace_inline, text)
    
    # Also handle $$...$$ if you have display math
    text = re.sub(r'\$\$(.*?)\$\$', replace_inline, text)
    
    return text

def create_gradio_interface():
    """Create the Gradio interface"""
    
    mathjax_head = """
    <script>
    window.MathJax = {
      tex: {
        inlineMath: [['\\\\(', '\\\\)']],
        displayMath: [['\\\\[', '\\\\]']],
        processEscapes: true
      }
    };
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
    """
    
    with gr.Blocks(theme=gr.themes.Soft(), title="Math Misconception Detector", head=mathjax_head) as demo:
        gr.Markdown("""
        # 🧮 Math Misconception Detection System
        
        This AI-powered system analyzes your mathematical reasoning to identify common misconceptions
        and provides personalized learning recommendations.
        
        **How it works:**
        1. Click "Start Assessment" to begin
        2. Answer each question and explain your reasoning
        3. Receive personalized feedback and learning resources
        """)
        
        start_btn = gr.Button("🚀 Start Assessment", variant="primary", size="lg")
        
        question_display = gr.HTML(label="Question")
        
        answer_radio = gr.Radio(
            choices=[],
            label="Select your answer:",
            visible=False
        )
        
        explanation_box = gr.Textbox(
            label="Explain your reasoning:",
            placeholder="Describe how you solved this problem...",
            lines=4,
            visible=False
        )
        
        submit_btn = gr.Button("Submit Answer", variant="primary", visible=False)
        
        feedback_display = gr.HTML(label="Feedback")
        
        # Full assessment button - hidden until assessment complete
        show_assessment_btn = gr.Button("📋 Show Full Assessment", variant="secondary", visible=False)
        assessment_table = gr.HTML(label="Full Assessment")
        
        start_btn.click(
            fn=start_assessment,
            inputs=[],
            outputs=[question_display, answer_radio, explanation_box, submit_btn, start_btn]
        )
        
        submit_btn.click(
            fn=submit_answer,
            inputs=[answer_radio, explanation_box],
            outputs=[
                question_display,
                feedback_display,
                answer_radio, 
                explanation_box, 
                submit_btn,
                show_assessment_btn  # Show button when assessment completes
            ]
        )
        
        show_assessment_btn.click(
            fn=show_full_assessment,
            inputs=[],
            outputs=[assessment_table]
        )
        
        gr.Markdown("""
        ---
        ### About This System
        - Powered by fine-tuned **Qwen-7B** with LoRA adapters
        - Analyzes **15 different problem types** (fractions, decimals, algebra, geometry, etc.)
        - Detects **35+ misconception patterns** using deep learning
        - Provides personalized recommendations with video tutorials
        
        **Note:** This is a research prototype for educational purposes.
        """)
    
    return demo


def assessment_inference_shared_base(grouped,
                                     kind='inference',
                                     correctness_checkpoint_path=None,
                                     misconception_checkpoint_path=None,
                                     model_name='Qwen/Qwen2-7B-Instruct',
                                     label_encoder_category=None,
                                     label_encoder_misconception=None,
                                     device=None,
                                     batch_size=16,
                                     max_length=450,
                                     device_type='gpu',
                                     add_ground_truth=False):
    """
    Complete assessment pipeline with DUAL model predictions using SHARED base model.
    This significantly reduces memory usage by loading the base Qwen model only once.
    """
    #grouped=generate_final_assessment(grouped,testing=add_ground_truth)
    # 1. Data preprocessing pipeline (same as before)
    augmenter = preprocessing_functions.TextAugmenter()
    
    grouped['StudentExplanation_Original']= grouped['StudentExplanation'] 
    grouped['StudentExplanation'] = grouped['StudentExplanation'].apply(augmenter.clean)
    grouped['QuestionText'] = grouped.apply(
        lambda x: preprocessing_functions.rephrase_question_and_explanation(x['QuestionText'])
        if x['problem_type'] == 7 else x['QuestionText'],
        axis=1
    )
    grouped['QuestionText'] = grouped['QuestionText'].apply(preprocessing_functions.strip_extra_whitespace)
    grouped['MC_Answer'] = grouped['MC_Answer'].apply(preprocessing_functions.standardize_answer_format_v5)
    grouped['Mathematical_Context'] = grouped[['QuestionText', 'MC_Answer', 'problem_type']].apply(
        lambda x: augment_functions.create_generic_math_context(x['QuestionText'], x['MC_Answer'], x['problem_type']),
        axis=1
    )
    grouped['StudentExplanation'] = grouped.apply(apply_feature_substitution, axis=1)
    grouped['Mathematical_Context_Reduced'] = grouped['Mathematical_Context'].apply(filter_math_context_case_insensitive)

    # 2. Model inference with SHARED base model
    if correctness_checkpoint_path is not None and misconception_checkpoint_path is not None:
        print("\n" + "="*80)
        print("LOADING SHARED BASE MODEL (MEMORY EFFICIENT)")
        print("="*80)

        # Auto-detect device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Load tokenizer
        print(f"\nLoading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = max_length

        # Setup quantization config
        if device_type == 'gpu':
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        else:
            bnb_config = None
            device = torch.device('cpu')

        # ========== LOAD SHARED BASE MODEL ==========
        print(f"\n📥 Loading SHARED Qwen base model (this will be reused)...")
        base_qwen = AutoModel.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map='auto' if device_type == 'gpu' else None
        )

        # Prepare for LoRA training
        base_qwen = prepare_model_for_kbit_training(base_qwen)

        lora_config = LoraConfig(
            r=64,
            lora_alpha=64,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        base_qwen = get_peft_model(base_qwen, lora_config)
        print(f"   ✅ Shared base model loaded")

        # ========== LOAD CORRECTNESS MODEL (with shared base) ==========
        print(f"\n📥 Loading CORRECTNESS model heads...")
        print(f"   Checkpoint: {correctness_checkpoint_path}")
        correctness_checkpoint = torch.load(
            correctness_checkpoint_path,
            map_location=device,
            weights_only=False
        )

        # Create wrapper WITHOUT loading base model again
        correctness_model = MultiTaskQwen.__new__(MultiTaskQwen)
        nn.Module.__init__(correctness_model)

        # Assign the shared base model
        correctness_model.qwen = base_qwen

        # Create classification heads
        hidden_size = base_qwen.config.hidden_size
        correctness_model.correctness_head = nn.Linear(hidden_size, 2)
        correctness_model.category_head = nn.Linear(hidden_size, len(label_encoder_category.classes_))
        correctness_model.misconception_head = nn.Linear(hidden_size, len(label_encoder_misconception.classes_))

        # Load checkpoint weights
        correctness_model.load_state_dict(correctness_checkpoint['model_state_dict'], strict=False)
        correctness_model.to(device)
        correctness_model.eval()

        print(f"   ✅ Loaded (Epoch {correctness_checkpoint.get('epoch', 'N/A')})")

        # ========== LOAD MISCONCEPTION MODEL (with shared base) ==========
        print(f"\n📥 Loading MISCONCEPTION model heads...")
        print(f"   Checkpoint: {misconception_checkpoint_path}")
        misconception_checkpoint = torch.load(
            misconception_checkpoint_path,
            map_location=device,
            weights_only=False
        )

        # Create wrapper WITHOUT loading base model again
        misconception_model = MultiTaskQwen.__new__(MultiTaskQwen)
        nn.Module.__init__(misconception_model)

        # Assign the SAME shared base model
        misconception_model.qwen = base_qwen

        # Create NEW classification heads (these will be different from correctness model)
        misconception_model.correctness_head = nn.Linear(hidden_size, 2)
        misconception_model.category_head = nn.Linear(hidden_size, len(label_encoder_category.classes_))
        misconception_model.misconception_head = nn.Linear(hidden_size, len(label_encoder_misconception.classes_))

        # Load checkpoint weights (will load the different head weights)
        misconception_model.load_state_dict(misconception_checkpoint['model_state_dict'], strict=False)
        misconception_model.to(device)
        misconception_model.eval()

        print(f"   ✅ Loaded (Epoch {misconception_checkpoint.get('epoch', 'N/A')})")
        print(f"\n⚡ Memory saved by sharing base model: ~7-8 GB")

        # 3. Run dual-model predictions
        print("\n" + "="*80)
        print("RUNNING DUAL-MODEL PREDICTIONS")
        print("="*80)

        grouped = add_dual_model_predictions(
            grouped=grouped,
            correctness_model=correctness_model,
            misconception_model=misconception_model,
            tokenizer=tokenizer,
            label_encoder_category=label_encoder_category,
            label_encoder_misconception=label_encoder_misconception,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            add_ground_truth=add_ground_truth
        )

        print("="*80)

    return grouped

def show_full_assessment() -> str:
    """Display the complete assessment as a table"""

    if state.results_df is None:
        return "<p>No assessment data available.</p>"
    
    html = """
    <div style="padding: 20px; font-family: 'Segoe UI', sans-serif; max-width: 900px; margin: auto;">
        <h2 style="color: #2c3e50;">📋 Full Assessment Summary</h2>
        <table style="width: 100%; border-collapse: collapse; margin-top: 20px;">
            <thead>
                <tr style="background: #3498db; color: white;">
                    <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">#</th>
                    <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">Question</th>
                    <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">Your Answer</th>
                    <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">Your Explanation</th>
                    <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">Result</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for i, row in state.results_df.iterrows():
        is_correct = str(row.get('predicted_correctness', '')).strip().lower() in ['correct', '1']
        result_text = "✅ Correct" if is_correct else "❌ Incorrect"
        row_bg = "#f0fff4" if is_correct else "#fff5f5"
        question_display = render_latex_to_html(str(row.get('QuestionText', 'N/A')))
        
        html += f"""
            <tr style="background: {row_bg};">
                <td style="padding: 12px; border: 1px solid #ddd; font-weight: bold;">{i + 1}</td>
                <td style="padding: 12px; border: 1px solid #ddd;">{question_display}</td>
                <td style="padding: 12px; border: 1px solid #ddd;">{row.get('MC_Answer', 'N/A')}</td>
                <td style="padding: 12px; border: 1px solid #ddd; font-style: italic;">"{row.get('StudentExplanation_Original', 'N/A')}"</td>
                <td style="padding: 12px; border: 1px solid #ddd; font-weight: bold;">{result_text}</td>
            </tr>
        """
    
    html += """
            </tbody>
        </table>
    </div>
    """
    
    return html
# ============================================================================
# LAUNCH
# ============================================================================

def run_app():
    gr.close_all()  # Close any existing instances
    demo = create_gradio_interface()
    demo.queue()
    
    # Detect environment and launch appropriately
    if 'SPACE_ID' in os.environ:
        # Running on HF Spaces
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False
        )
    else:
        # Running in Colab or local
        demo.launch(share=True)

if __name__ == "__main__":
    run_app()
