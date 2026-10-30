import re
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

# ================= INTENT EXAMPLES (no "general" bank) =================
INTENT_EXAMPLES = {
    "exit": [
        "exit", "quit", "stop", "bye", "goodbye",
        "bye friday", "goodbye friday", "shut down",
        "turn yourself off", "close the assistant",
        "please stop", "I'm done", "we're done here",
        "see you later", "that's all", "see ya", "later friday",
        "talk to you later", "that will be all", "go to sleep",
        "shut yourself down", "I'm leaving now"
    ],
    "greeting": [
        "hello", "hi", "hey", "hello friday", "hi friday",
        "hey friday", "good morning", "good afternoon",
        "good evening", "yo", "sup", "what's up",
        "are you there", "you there", "hello again",
        "hey there", "howdy", "yo friday", "hiya"
    ],
    "time": [
        "what time is it", "what's the time", "tell me the time",
        "what is the current time", "what time is it now",
        "do you know what time it is", "any idea what time it is",
        "could you tell me the time", "wats the time",
        "time please", "what does the clock say"
    ],
    "date": [
        "what's today's date", "what is today's date",
        "what is the date today", "tell me today's date",
        "what day is it", "what day is it today",
        "what day of the week is it", "what month is it",
        "what year is it", "wat day is it", "what's the date"
    ],
}

# Minimum similarity needed to accept each intent.
# Below this, the message is treated as "general".
# Exit is strict because a false positive shuts FRIDAY down.
INTENT_MIN = {
    "exit": 0.70,
    "greeting": 0.60,
    "time": 0.60,
    "date": 0.60,
}

# ================= STATE EXAMPLES =================
STATE_EXAMPLES = {
    "idle": [
        "okay", "alright", "never mind", "forget it",
        "nothing for now", "I'm good", "that's fine",
        "wait", "hold on"
    ],
    "casual_chat": [
        "how are you", "how are you doing", "what's up",
        "let's chat", "let's just talk", "tell me something funny",
        "do you have feelings", "are you there", "yo friday",
        "good morning", "hello again",
        # complaints / corrections / reactions
        "I didn't ask for that", "that's not what I asked",
        "that's not what I meant", "you misunderstood me",
        "you're wrong", "that's incorrect", "why did you say that",
        "stop making things up", "that doesn't make sense",
        "thanks", "thank you", "nice", "cool", "that's funny",
        "I see", "really", "wow"
    ],
    "storytelling": [
        "tell me a story", "write me a story",
        "tell me another story", "once upon a time",
        "make the ending sad", "make the ending happier",
        "change the ending", "add a dragon to the story",
        "change the character", "make the story longer",
        "add more detail to the story", "tell me a bedtime story"
    ],
    "coding": [
        "help me debug my code", "help me debug my Python code",
        "why is my program crashing", "why is my code not working",
        "what is wrong with my code", "help me fix this bug",
        "explain this error", "what does this error mean",
        "write a Python script", "write a Python function",
        "how do I use a for loop", "why is my variable undefined",
        "what does this traceback mean", "my code keeps failing"
    ],
    "planning": [
        "help me plan my career", "what should I learn next",
        "give me a roadmap", "help me make a schedule",
        "what steps should I take", "plan my week",
        "how do I become a data scientist", "help me set goals",
        "what should I do first", "help me prioritize my tasks",
        "make a study plan", "I don't know where to start"
    ],
    "question_answering": [
        "what is Python", "how does a CPU work",
        "explain machine learning", "what is artificial intelligence",
        "who invented the telephone", "what is the capital of France",
        "how does photosynthesis work", "what causes earthquakes",
        "what is a binary tree", "how does the internet work",
        "what is blockchain", "explain neural networks"
    ],
}

STATE_MIN = 0.45  # below this, keep the previous state

# Short follow-ups that mean "stay in whatever we were doing"
CONTINUATION_PHRASES = {
    "continue", "continue it", "keep going", "go on", "more",
    "next", "tell me more", "what happened next", "and then",
    "then what", "go ahead", "finish it", "continue from there",
    "and then what happened", "tell me what happens next"
}

# ================= SETUP =================
def build_bank(examples):
    return {
        label: model.encode(sentences, convert_to_tensor=True)
        for label, sentences in examples.items()
    }

INTENT_BANK = build_bank(INTENT_EXAMPLES)
STATE_BANK = build_bank(STATE_EXAMPLES)

def normalize(text):
    """Lowercase, strip punctuation (Whisper adds it), collapse spaces."""
    text = re.sub(r"[^a-z0-9' ]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()

def score_bank(embedding, bank):
    """Best cosine similarity per label."""
    return {
        label: float(util.cos_sim(embedding, examples)[0].max())
        for label, examples in bank.items()
    }

# ================= INTENT =================
def classify_intent(embedding):
    scores = score_bank(embedding, INTENT_BANK)
    best = max(scores, key=scores.get)

    if scores[best] >= INTENT_MIN[best]:
        return best, scores[best]

    return "general", scores[best]

# ================= STATE =================
def classify_state(embedding, clean_text, previous_state):
    if clean_text in CONTINUATION_PHRASES:
        return previous_state, 1.0

    scores = score_bank(embedding, STATE_BANK)
    best = max(scores, key=scores.get)
    best_score = scores[best]

    # Weak match: don't switch topics on weak evidence
    if best_score < STATE_MIN:
        return previous_state, best_score

    # Stay in the previous state if it's nearly as good as the winner
    if previous_state != "idle":
        prev_score = scores.get(previous_state, -1)
        if prev_score >= best_score - 0.05:
            return previous_state, prev_score

    return best, best_score

# ================= COMBINED =================
def classify_both(text, previous_state):
    clean = normalize(text)
    embedding = model.encode(clean, convert_to_tensor=True)  # encode once

    intent, intent_score = classify_intent(embedding)

    if intent in {"exit", "time", "date"}:
        state, state_score = "idle", 1.0
    elif intent == "greeting":
        state, state_score = "casual_chat", 1.0
    else:
        state, state_score = classify_state(embedding, clean, previous_state)

    return intent, round(intent_score, 4), state, round(state_score, 4)