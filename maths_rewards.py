import math, regex as re
from fractions import Fraction
from sympy import sympify
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application, convert_xor

TRANSFORMS = (standard_transformations + (implicit_multiplication_application, convert_xor))

NUM_PAT = re.compile(r'[-+]?(\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?%?')

ASSISTANT_PREFIX = "<|im_start|>assistant\n"

def _strip_commas(x: str) -> str:
    return x.replace(",", "")

def _to_number(token: str):
    """
    Accepts forms like 1,234, 12.5, 3/4, 12.5%, -7, etc.
    Returns float (or int when exact), else None.
    """
    s = token.strip()
    is_percent = s.endswith('%')
    if is_percent:
        s = s[:-1]
    # fraction?
    if '/' in s and not re.search(r'[a-zA-Z]', s):
        try:
            val = float(Fraction(_strip_commas(s)))
            return val / 100.0 if is_percent else val
        except Exception:
            pass
    try:
        val = float(_strip_commas(s))
        return val / 100.0 if is_percent else val
    except Exception:
        return None

def _extract_final_answer(text: str):
    """
    First try to extract from formatting '#### <answer>'
    Fallback by extracting the last numeric-looking token in the output.
    """
    # #### answer
    m = re.search(r'####\s*([^\n]+)', text)
    if m:
        cand = m.group(1).strip()
        # try a number directly
        n = _to_number(cand)
        if n is not None:
            return n
        # try parsing an expression
        try:
            val = float(sympify(cand, transformations=TRANSFORMS))
            return val
        except Exception:
            pass

    # last numeric token
    nums = list(NUM_PAT.finditer(text))
    if nums:
        token = nums[-1].group(0)
        n = _to_number(token)
        if n is not None:
            return n

    return None

def numeric_equal(a, b, rel_tol=1e-6, abs_tol=1e-6):
    if a == b:  # handles exact ints
        return True
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

def gsm8k_numeric_reward(gold_answer_str: str, model_output: str,
                         rel_tol=1e-6, abs_tol=1e-6):
    """
    Returns a scalar in [0,1].
    Base: 1.0 if numeric correct, else 0.0
    Shaping: +0.05 if uses '#### <answer>' format; -0.2 if no numeric found.
    Clipped to [0,1].
    """

    gold = _extract_final_answer(gold_answer_str)

    if gold is None:
        print("This shouldn't have happened....")

    pred = _extract_final_answer(model_output)
    shaped = 0.0

    if re.search(r'####\s*\d+[^\n]+', model_output):
        shaped += 0.05  # tiny nudge toward consistent formatting

    if pred is None or gold is None:
        return max(0.0, min(1.0, 0.0 + shaped - 0.2))  # penalise missing numeric

    base = 1.0 if numeric_equal(pred, gold, rel_tol=rel_tol, abs_tol=abs_tol) else 0.0
    return max(0.0, min(1.0, base + shaped))
