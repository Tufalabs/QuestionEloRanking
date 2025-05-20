"""
Elo rating system for questions.
"""
import os
import json
import random
import time
import asyncio

from utils.inference import generate_text

# Default Elo rating for new questions
DEFAULT_RATING = 1200
# File paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
QUESTIONS_FILE = os.path.join(PROJECT_ROOT, 'merged_questions.json')
RATINGS_FILE = os.path.join(PROJECT_ROOT, 'question_elos.json')

def load_questions():  # -> Dict[str, str]
    """
    Load questions from the merged JSON file.
    Returns a dict mapping question_id to question text.
    """
    with open(QUESTIONS_FILE, 'r') as f:
        data = json.load(f)
    questions = {}
    for q in data.get('questions', []):
        qid = q.get('id')
        text = q.get('question')
        if qid and text:
            questions[qid] = text
    return questions

def load_ratings():  # -> Dict[str, dict]
    """
    Load or initialize Elo ratings for questions, including uncertainty tracking.
    Returns a dict mapping question_id to a dict with:
      - rating (int): current Elo rating
      - matches (int): number of pairwise comparisons participated in
      - uncertainty (float | None): 1/sqrt(matches) or None if no data
    """
    if os.path.exists(RATINGS_FILE):
        with open(RATINGS_FILE, 'r') as f:
            raw = json.load(f)
    else:
        raw = {}
    questions = load_questions()
    ratings = {}
    updated = False
    for qid in questions:
        entry = raw.get(qid)
        up = None
        if isinstance(entry, dict):
            r = entry.get('rating', DEFAULT_RATING)
            m = entry.get('matches', 0)
            u = entry.get('uncertainty')
            up = entry.get('uncertainty_points')
        elif isinstance(entry, (int, float)):
            r = entry
            m = 0
            u = None
            updated = True
        else:
            r = DEFAULT_RATING
            m = 0
            u = None
            updated = True
        ratings[qid] = {
            'rating': int(r),
            'matches': int(m),
            'uncertainty': u,
            'uncertainty_points': up
        }
    if updated:
        save_ratings(ratings)
    return ratings

def save_ratings(ratings):
    """
    Save Elo ratings to the JSON file.
    """
    with open(RATINGS_FILE, 'w') as f:
        json.dump(ratings, f, indent=2, sort_keys=True)

async def rank_questions(question_ids, model='o4-mini'):
    """
    Use an LLM to rank the given questions from easiest to hardest.
    Returns a list of question_ids in ranked order.
    """
    questions = load_questions()
    # Build prompt
    prompt = [
        "You are given a list of math questions."
        " Please rank them from easiest to hardest."
        " Respond with ONLY a JSON array of question IDs in order from easiest to hardest, with no additional text."
        " Here are the questions:"
    ]
    for idx, qid in enumerate(question_ids, start=1):
        text = questions.get(qid, '').replace('\n', ' ')
        prompt.append(f"{idx}. (ID: {qid}) {text}")
    full_prompt = '\n'.join(prompt)
    # Call LLM
    try:
        resp = await generate_text(model, full_prompt, max_tokens=1024, temperature=0)
        # Clean markdown code fences if present
        resp_str = resp.strip()
        if resp_str.startswith('```') and resp_str.endswith('```'):
            lines = resp_str.splitlines()
            # drop the first and last fence lines
            resp_str = '\n'.join(lines[1:-1])
        # Parse JSON array
        ranking = json.loads(resp_str)
        # Map any numeric entries (ints or numeric strings) to question IDs by position
        mapped = []
        for item in ranking:
            idx = None
            if isinstance(item, int):
                idx = item
            elif isinstance(item, str) and item.isdigit():
                idx = int(item)
            if idx is not None and 1 <= idx <= len(question_ids):
                mapped.append(question_ids[idx - 1])
            else:
                mapped.append(item)
        ranking = mapped
        # Validate that returned IDs match the requested set
        if not isinstance(ranking, list) or set(ranking) != set(question_ids):
            raise ValueError(f"Invalid ranking response: {resp}")
        return ranking
    except Exception as e:
        print(f"Error ranking questions: {e}")
        print(f"LLM response: {resp if 'resp' in locals() else None}")
        return None

def update_ratings(ratings, question_ids, ranking, k_factor=32):
    """
    Update Elo ratings based on the ranking of a batch of questions.

    The K-factor controls how much ratings can change in each batch: it is the total scaling
    factor distributed across all pairwise comparisons. A larger k_factor results in more
    volatile ratings (larger adjustments), while a smaller k_factor yields more conservative
    updates. Internally, each question-to-question comparison within the batch uses k_factor/(n-1).

    Parameters:
      ratings (dict): current ratings mapping question_id to rating.
      question_ids (list): list of question_ids in this batch.
      ranking (list): question_ids sorted from easiest to hardest (new order).
      k_factor (float): total K-factor per batch (default: 32).

    Returns:
      dict: updated ratings mapping question_id to new integer rating.
    """
    # Snapshot of old ratings
    old = {qid: ratings[qid]['rating'] for qid in question_ids}
    n = len(question_ids)
    if n < 2:
        return ratings
    # Per-match K
    k_per_match = k_factor / (n - 1)
    # Compute expected and actual scores
    for pos, qid in enumerate(ranking):
        R_i = old[qid]
        # Actual score: wins against all easier questions (ranked before)
        S_i = pos
        # Expected score: sum of expected vs each other
        E_i = 0.0
        for other in question_ids:
            if other == qid:
                continue
            R_j = old[other]
            E_i += 1 / (1 + 10 ** ((R_j - R_i) / 400))
        # Update rating
        new_R = R_i + k_per_match * (S_i - E_i)
        ratings[qid]['rating'] = int(round(new_R))
        # Increment match count (each batch gives n-1 pairwise comparisons)
        ratings[qid]['matches'] += n - 1
        # Recompute uncertainty as 1/sqrt(matches)
        m = ratings[qid]['matches']
        # Uncertainty ratio (0-1): ~1/sqrt(matches)
        u = (1.0 / m**0.5) if m > 0 else None
        ratings[qid]['uncertainty'] = u
        # Uncertainty in Elo points (approximate rating error): scale ratio by 400
        ratings[qid]['uncertainty_points'] = (u * 400) if u is not None else None
    return ratings

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Elo rating for questions")
    parser.add_argument('--batch-size', '-b', type=int, default=20,
                        help='Number of questions to rank per iteration')
    parser.add_argument('--num-questions-per-iteration', '-q', type=int,
                        dest='batch_size',
                        help='Alias for --batch-size')
    parser.add_argument('--iterations', '-n', type=int, default=1,
                        help='Number of ranking iterations')
    parser.add_argument('--model', '-m', type=str, default='gpt-4o-mini',
                        help='LLM model to use for ranking')
    parser.add_argument('--k-factor', '-k', type=float, default=32,
                        help='Total K-factor per batch (controls volatility of rating changes; per-pair K = k_factor/(n-1))')
    parser.add_argument('--sleep', '-s', type=float, default=0,
                        help='Seconds to sleep between iterations')
    args = parser.parse_args()
    # Load or init ratings
    ratings = load_ratings()
    all_ids = list(ratings.keys())
    for i in range(args.iterations):
        batch = random.sample(all_ids, min(args.batch_size, len(all_ids)))
        ranking = asyncio.run(rank_questions(batch, model=args.model))
        if not ranking:
            print('Skipping iteration due to ranking error.')
            continue
        ratings = update_ratings(ratings, batch, ranking, k_factor=args.k_factor)
        save_ratings(ratings)
        print(f'Iteration {i+1}/{args.iterations} complete. Ratings updated.')
        if args.sleep and i < args.iterations - 1:
            time.sleep(args.sleep)

if __name__ == '__main__':  # pragma: no cover
    main()