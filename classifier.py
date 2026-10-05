from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

INTENT_EXAMPLES = {
    "exit": [
        "exit", "quit", "stop", "bye", "goodbye",
        "bye friday", "goodbye friday", "shut down",
        "turn yourself off", "close the assistant",
        "please stop", "I'm done", "we're done here" ,"see you later",
        "that's all"
    ],
    "greeting": [
        "hello", "hi", "hey", "hello friday", "hi friday",
        "hey friday", "good morning", "good afternoon",
        "good evening", "yo", "sup", "what's up",
        "are you there", "you there", "hello again"
    ],
    "time": [
        "what time is it", "what's the time", "tell me the time",
        "what is the current time", "what time is it now",
        "do you know what time it is", "any idea what time it is",
        "could you tell me the time", "wats the time"
    ],
    "date": [
        "what's today's date", "what is today's date",
        "what is the date today", "tell me today's date",
        "what day is it", "what day is it today",
        "what day of the week is it", "what month is it",
        "what year is it", "wat day is it"
    ],
    "general": [
        "explain this to me", "help me understand something",
        "what is machine learning", "what is Python",
        "help me fix my code", "tell me a story",
        "help me plan my career", "why is the sky blue",
        "how does a CPU work", "tell me something interesting",
        "continue the story",
        "continue",
        "keep going",
        "what happened next",
        "make the ending happier",
        "change the ending",
        "tell me what happens next",
    ]
}

STATE_EXAMPLES = {
    "idle": [
        "okay", "alright", "never mind", "forget it",
        "nothing for now", "I'm good", "that's fine",
        "that's all", "wait", "hold on"
    ],
    "casual_chat": [
        "how are you", "how are you doing", "what's up",
        "let's chat", "let's just talk", "tell me something funny",
        "do you have feelings", "are you there", "yo friday",
        "good morning", "hello again"
    ],
    "storytelling": [
        "tell me a story", "write me a story",
        "continue the story", "continue it", "keep going",
        "what happened next", "and then what happened",
        "make the ending sad", "make the ending happier",
        "add a dragon", "change the character",
        "make it longer", "add more detail",
        "tell me more of the story"
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
    ]
}

def build_bank(examples):
    return {
        label: model.encode(sentences, convert_to_tensor=True)
        for label, sentences in examples.items()
    }

INTENT_BANK = build_bank(INTENT_EXAMPLES)
STATE_BANK = build_bank(STATE_EXAMPLES)

def classify(text, bank):
    embedding = model.encode(text, convert_to_tensor=True)
    results = {}
    for label, examples in bank.items():
        scores = util.cos_sim(embedding, examples)[0]
        results[label] = float(scores.max())
    best = max(results, key=results.get)
    return best, results[best], results

CONTINUATION_PHRASES = {
    "continue", "continue it", "keep going", "go on",
    "more", "next", "tell me more", "what happened next",
    "and then", "then what", "go ahead", "finish it",
    "continue from there"
}

def classify_state(text, previous_state):
    text_clean = text.strip().lower()
    if "help me plan" in text_clean or "plan my carrer" in text_clean:
        return "planning",1.0

    if text_clean in CONTINUATION_PHRASES:
        return previous_state, 1.0

    state, score, scores = classify(text, STATE_BANK)

    previous_score = scores.get(previous_state, -1)

    # Only keep the previous state when it is genuinely competitive.
    # A clearly stronger new state wins.
    if previous_state != "idle":
        if previous_score >= score - 0.05:
            state = previous_state
            score = previous_score

    return state, round(score, 4)

def classify_both(text, previous_state):
    intent, intent_score, _ = classify(text, INTENT_BANK)

    # Certain intents should always leave FRIDAY in idle state
    if intent in {"exit", "time", "date"}:
        state = "idle"
        state_score = 1.0

    # Continuation requests belong to the current conversation state
    elif text.strip().lower() in CONTINUATION_PHRASES:
        state = previous_state
        state_score = 1.0

    else:
        state, state_score = classify_state(text, previous_state)

    return intent, round(intent_score, 4), state, round(state_score, 4)