"""
FRIDAY - Standalone Classifier Test
====================================
Tests Intent + Conversation State classification
using sentence embeddings (no LLM, no API calls).

Model : all-MiniLM-L6-v2  (~22 MB)
Speed : ~5ms per classification on CPU
"""

from sentence_transformers import SentenceTransformer, util
import torch

# ─────────────────────────────────────────────
# 1. LOAD MODEL
# ─────────────────────────────────────────────
print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model ready.\n")


# ─────────────────────────────────────────────
# 2. INTENT EXAMPLES
#    Each label has a bank of example sentences.
#    The classifier finds which bank is closest
#    in meaning to the user's input.
# ─────────────────────────────────────────────
INTENT_EXAMPLES = {
    "exit": [
        "exit",
        "stop",
        "quit",
        "bye friday",
        "goodbye friday",
        "shut down",
        "turn yourself off",
        "I want you to stop",
        "friday please stop",
        "I'm done for today",
        "close the assistant",
        "we're done here",
        "that's all for now goodbye",
    ],
    "greeting": [
        "hello",
        "hi friday",
        "hey there",
        "good morning friday",
        "good evening",
        "what's up",
        "hey how are you",
        "howdy",
        "greetings",
        "yo friday",
        "hi there how are you doing",
        "good afternoon",
    ],
    "time": [
        "what time is it",
        "what's the time",
        "tell me the time",
        "what is the current time",
        "can you tell me the time please",
        "what time is it right now",
        "do you know what time it is",
        "give me the current time",
    ],
    "date": [
        "what's today's date",
        "what day is it today",
        "tell me today's date",
        "what is the date today",
        "what is today's date",
        "could you tell me the date",
        "what day of the week is it",
        "what month is it",
        "what year is it",
    ],
    "general": [
        "explain this to me",
        "tell me about artificial intelligence",
        "what is machine learning",
        "help me understand quantum physics",
        "who is the president of France",
        "what is the capital of Japan",
        "how does a black hole form",
        "write me a story",
        "help me fix my Python code",
        "what should I eat for dinner",
        "tell me something interesting",
        "can you help me plan my week",
        "why is the sky blue",
        "what is the speed of light",
    ],
}


# ─────────────────────────────────────────────
# 3. CONVERSATION STATE EXAMPLES
#    Same approach — example bank per state.
#    The previous_state is injected as a
#    soft bias (explained below).
# ─────────────────────────────────────────────
STATE_EXAMPLES = {
    "idle": [
        "okay",
        "alright",
        "never mind",
        "forget it",
        "nothing for now",
        "I'm good thanks",
        "no that's fine",
        "stop",
        "wait",
    ],
    "casual_chat": [
        "how are you doing",
        "what do you think about life",
        "do you have feelings",
        "are you happy",
        "tell me something funny",
        "let's just chat",
        "what's your favorite thing",
        "do you get bored",
        "you seem smart",
        "I like talking to you",
    ],
    "storytelling": [
        "tell me a story",
        "write a short story",
        "once upon a time",
        "continue the story",
        "what happened next",
        "continue it",
        "make the ending sad",
        "add a dragon to the story",
        "change the character",
        "tell another story",
        "make it longer",
        "add more detail to the story",
        "continue from where you left off",
        "tell me more of the story",
    ],
    "coding": [
        "help me debug my Python code",
        "why is my program crashing",
        "explain this error message",
        "fix this function for me",
        "write a Python script",
        "how do I use a for loop",
        "what is wrong with my code",
        "help me with this bug",
        "write a class in Python",
        "how does recursion work",
        "why is my variable undefined",
        "what does this stack trace mean",
        "write a function that sorts a list",
        "how do I read a file in Python",
    ],
    "planning": [
        "help me plan my career",
        "what should I learn next",
        "give me a roadmap",
        "help me make a schedule",
        "what steps should I take",
        "plan my week for me",
        "how do I become a data scientist",
        "what is the best order to learn things",
        "help me set goals",
        "I want to start a project",
        "how should I structure my learning",
        "what should I do first",
        "help me prioritize my tasks",
        "make a study plan for me",
    ],
    "question_answering": [
        "what is Python",
        "how does a CPU work",
        "explain machine learning to me",
        "what is artificial intelligence",
        "who invented the telephone",
        "what is the capital of France",
        "how does photosynthesis work",
        "what causes earthquakes",
        "define recursion",
        "what is the difference between RAM and ROM",
        "how does the internet work",
        "what is blockchain",
        "explain neural networks",
        "what is a binary tree",
    ],
}


# ─────────────────────────────────────────────
# 4. PRE-COMPUTE EMBEDDINGS
#    We embed all example sentences once at
#    startup so classification is instant.
# ─────────────────────────────────────────────
def build_embedding_bank(examples: dict) -> dict:
    """
    Pre-computes embeddings for every example sentence.
    Returns a dict of { label: tensor_of_embeddings }
    """
    bank = {}
    for label, sentences in examples.items():
        embeddings = model.encode(sentences, convert_to_tensor=True)
        bank[label] = embeddings
    return bank


print("Building embedding banks...")
INTENT_BANK = build_embedding_bank(INTENT_EXAMPLES)
STATE_BANK  = build_embedding_bank(STATE_EXAMPLES)
print("Banks ready.\n")


# ─────────────────────────────────────────────
# 5. CORE CLASSIFIER
# ─────────────────────────────────────────────
def classify(text: str, bank: dict) -> tuple[str, float]:
    """
    Embeds the input text, compares it against every
    label's example bank using cosine similarity,
    and returns the best label + confidence score.

    Returns:
        (label, confidence)  e.g. ("coding", 0.82)
    """
    input_embedding = model.encode(text, convert_to_tensor=True)

    best_label      = None
    best_score      = -1.0

    for label, example_embeddings in bank.items():
        # Compare input against all examples for this label.
        # scores shape: (num_examples,)
        scores = util.cos_sim(input_embedding, example_embeddings)[0]

        # Take the single highest-scoring example for this label.
        top_score = float(scores.max())

        if top_score > best_score:
            best_score = top_score
            best_label = label

    return best_label, round(best_score, 4)


# ─────────────────────────────────────────────
# 6. CONVERSATION STATE WITH PREVIOUS-STATE BIAS
#    When the user says something ambiguous like
#    "continue" or "what next?", we want to stay
#    in the current state rather than randomly jump.
#    We do this by boosting the current state's
#    score slightly before comparing.
# ─────────────────────────────────────────────
CONTINUITY_BIAS = 0.12   # how much to favour staying in current state
CONTINUITY_WORDS = {     # short words that almost always mean "continue"
    "continue", "go on", "keep going", "more", "next",
    "continue it", "tell me more", "what happened", "and then",
    "finish it", "go ahead", "proceed",
}

def classify_state(
    text: str,
    previous_state: str
) -> tuple[str, float]:
    """
    Classifies conversation state, with a bias
    toward keeping the previous state for
    ambiguous or continuation-style messages.

    Returns:
        (state, confidence)
    """
    input_lower = text.strip().lower()

    # Hard rule: very short continuation phrases
    # almost always mean "stay in current state"
    if input_lower in CONTINUITY_WORDS:
        return previous_state, 1.0

    input_embedding = model.encode(text, convert_to_tensor=True)

    best_label = None
    best_score = -1.0

    for label, example_embeddings in STATE_BANK.items():
        scores    = util.cos_sim(input_embedding, example_embeddings)[0]
        top_score = float(scores.max())

        # Apply continuity bias to the previous state
        if label == previous_state:
            top_score += CONTINUITY_BIAS

        if top_score > best_score:
            best_score = top_score
            best_label = label

    # Cap displayed confidence at 1.0 after bias
    displayed_confidence = round(min(best_score, 1.0), 4)

    return best_label, displayed_confidence


# ─────────────────────────────────────────────
# 7. AUTOMATED TEST SUITE
#    Run 20+ varied sentences and print results.
#    Expected values let you see where it's wrong.
# ─────────────────────────────────────────────
TEST_CASES = [
    # (input, expected_intent, prev_state, expected_state)

    # --- INTENT: greeting ---
    ("hey friday",                  "greeting",  "idle",         "casual_chat"),
    ("good morning",                "greeting",  "idle",         "casual_chat"),
    ("yo what's up",                "greeting",  "idle",         "casual_chat"),

    # --- INTENT: time ---
    ("what time is it?",            "time",      "idle",         "idle"),
    ("could you tell me the time?", "time",      "idle",         "idle"),
    ("what's the current time",     "time",      "idle",         "idle"),

    # --- INTENT: date ---
    ("what's today's date?",        "date",      "idle",         "idle"),
    ("what day is it today",        "date",      "idle",         "idle"),

    # --- INTENT: exit ---
    ("bye friday",                  "exit",      "idle",         "idle"),
    ("please shut yourself down",   "exit",      "idle",         "idle"),
    ("I'm done, close down",        "exit",      "idle",         "idle"),

    # --- INTENT: general / STATE: storytelling ---
    ("tell me a story",             "general",   "idle",         "storytelling"),
    ("continue the story",          "general",   "storytelling", "storytelling"),
    ("make the ending happier",     "general",   "storytelling", "storytelling"),
    ("add more detail",             "general",   "storytelling", "storytelling"),

    # --- INTENT: general / STATE: coding ---
    ("help me fix my Python code",  "general",   "idle",         "coding"),
    ("why is my program crashing",  "general",   "coding",       "coding"),
    ("what does this error mean",   "general",   "coding",       "coding"),

    # --- INTENT: general / STATE: planning ---
    ("help me plan my career",      "general",   "idle",         "planning"),
    ("what should I learn next",    "general",   "planning",     "planning"),

    # --- INTENT: general / STATE: question_answering ---
    ("what is machine learning",    "general",   "idle",         "question_answering"),
    ("how does a CPU work",         "general",   "idle",         "question_answering"),

    # --- INTENT: general / STATE: casual_chat ---
    ("do you ever get bored",       "general",   "casual_chat",  "casual_chat"),

    # --- STATE SWITCH: storytelling → coding ---
    ("now help me debug my code",   "general",   "storytelling", "coding"),

    # --- STATE SWITCH: coding → storytelling ---
    ("forget the code, tell me a story", "general", "coding",    "storytelling"),
]

def run_tests():
    print("=" * 60)
    print("AUTOMATED TEST SUITE")
    print("=" * 60)

    intent_correct = 0
    state_correct  = 0
    total          = len(TEST_CASES)

    for text, exp_intent, prev_state, exp_state in TEST_CASES:
        intent, intent_conf = classify(text, INTENT_BANK)
        state,  state_conf  = classify_state(text, prev_state)

        intent_ok = "✓" if intent == exp_intent else "✗"
        state_ok  = "✓" if state  == exp_state  else "✗"

        if intent == exp_intent:
            intent_correct += 1
        if state == exp_state:
            state_correct += 1

        print(f'\n Input : "{text}"')
        print(f'  Intent : {intent_ok} {intent:<22} (expected: {exp_intent}) conf={intent_conf}')
        print(f'  State  : {state_ok}  {state:<22} (prev: {prev_state}, expected: {exp_state}) conf={state_conf}')

    print("\n" + "=" * 60)
    print(f"Intent accuracy : {intent_correct}/{total} ({100*intent_correct//total}%)")
    print(f"State  accuracy : {state_correct}/{total}  ({100*state_correct//total}%)")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────
# 8. INTERACTIVE TERMINAL LOOP
# ─────────────────────────────────────────────
def interactive_loop():
    print("=" * 60)
    print("FRIDAY Classifier — Interactive Test")
    print("Type a message. Type 'quit' to exit.")
    print("=" * 60 + "\n")

    previous_state = "idle"

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Exiting interactive test.")
            break

        intent, intent_conf = classify(user_input, INTENT_BANK)
        state,  state_conf  = classify_state(user_input, previous_state)

        print(f"\n  Intent            : {intent}")
        print(f"  Intent Confidence : {intent_conf}")
        print(f"  Conversation State: {state}")
        print(f"  State Confidence  : {state_conf}")
        print(f"  (previous state was: {previous_state})\n")

        previous_state = state


# ─────────────────────────────────────────────
# 9. ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_tests()
    interactive_loop()
    