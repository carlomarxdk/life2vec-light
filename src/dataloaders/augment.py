"""
This script contains the data augmentation functions that alter the input data during the training procedure.
"""
import random
import numpy as np
from .types import PersonDocument


def drop_tokens(document: PersonDocument, p) -> PersonDocument:
    """Randomly drop tokens from a sentence"""
    for sentence in document.sentences:
        if np.random.rand(1) < p:
            sentence.pop(random.randrange(len(sentence)))
    return document


def align_document(document: PersonDocument) -> PersonDocument:
    """Aligns the document with timecut_pos"""

    cut = document.timecut_pos
    document.sentences = document.sentences[:cut]
    document.abspos = document.abspos[:cut]
    document.age = document.age[:cut]
    return document


def make_timecut(
    document: PersonDocument,
    min_abspos: int = 732,  # we do not cut events too early in the sequence.
) -> PersonDocument:  # 732 corresponds to time between 01/01/08-01/01/10
    """Randomly Right Cut the Document"""

    num_events = len(document.sentences)
    # We do not cut if too few events
    if num_events <= 12:
        return document

    # Determine a random cut position
    timecut_pos = np.random.randint(low=12, high=int(num_events), size=1)[0]

    # INCLUDE THE WHOLE "DAY"
    while True:
        data_check = min_abspos <= document.abspos[timecut_pos]
        if timecut_pos + 1 >= len(document.abspos):
            break
        elif (
            document.abspos[timecut_pos] == document.abspos[timecut_pos + 1]
            or not data_check
        ):
            timecut_pos += 1
        else:
            break

    document.sentences = document.sentences[:timecut_pos]
    document.abspos = document.abspos[:timecut_pos]
    document.age = document.age[:timecut_pos]

    return document


def add_noise2time(document: PersonDocument) -> PersonDocument:
    """Add noise to the Absolute Position Values, adds +/- 5 days to each ABSPOS value. 
    You could potentially do the same for the age."""
    noise = np.random.randint(low=-5, high=5, size=len(document.abspos))
    document.abspos = [
        np.clip(pos + noise[i], a_min=0, a_max=4329)
        for i, pos in enumerate(document.abspos)
    ]
    return document


def resample_document(document: PersonDocument) -> PersonDocument:
    """Randomly remove events from the document"""
    num_records = len(
        document.sentences)  # Total number of the EVENTS in the sequence; we call EVENTS as SENTENCES
    num_to_remove = np.random.randint(
        low=int(np.floor(num_records * 0.25)),
        # Make sure that we remove at most 50% of the events
        high=int(np.floor((num_records * 0.5))),
        size=1,
    )
    idx_to_remove = set(
        np.random.choice(np.arange(0, num_records),
                         size=num_to_remove, replace=False)
    )

    document.sentences = [
        # remove the sentences
        i for idx, i in enumerate(document.sentences) if idx not in idx_to_remove
    ]
    document.abspos = [
        # remove associated abspos
        i for idx, i in enumerate(document.abspos) if idx not in idx_to_remove
    ]
    document.age = [i for idx, i in enumerate(  # remove associated age
        document.age) if idx not in idx_to_remove]

    # The first sentence in the document should start with [0,1]
    if document.abspos[0] != 1:
        offset = document.abspos[0] - 1
        document.abspos = [np.maximum(0, i - offset) for i in document.abspos]

    return document


def shuffle_sentences(document: PersonDocument) -> PersonDocument:
    """Shuffles the order of tokens in each sentence in the document.
    To make sure that the order of tokens within the sentence/event does not matter.
    In theory, this is a redundant step.
    """

    document.shuffled = True
    order = list(range(0, len(document.abspos)))
    random.shuffle(order)
    order = order
    document.sentences = [document.sentences[i] for i in order]
    if document.segment is not None:
        document.segment = [document.segment[i] for i in order]
    document.abspos = [document.abspos[i] for i in order]
    document.age = [document.age[i] for i in order]

    return document
