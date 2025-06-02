"""
Elo rating system for questions.
"""
import os
import json
import random
import time
import asyncio
import statistics
from datetime import datetime

from utils.inference import generate_text

# Default Elo rating for new questions
DEFAULT_RATING = 1200
# File paths
PROJECT_ROOT = os.path.dirname(__file__)
QUESTIONS_FILE = os.path.join(PROJECT_ROOT, 'questions.json')
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
            data = json.load(f)
        # Handle both old format (direct ratings) and new format (with statistics)
        if 'ratings' in data:
            raw = data['ratings']
        else:
            raw = data
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

def calculate_stats(ratings):
    """
    Calculate statistics for the ratings.
    """
    if not ratings:
        return {}
    
    rating_values = [r['rating'] for r in ratings.values()]
    match_counts = [r['matches'] for r in ratings.values()]
    
    stats = {
        'total_questions': len(ratings),
        'average_rating': round(statistics.mean(rating_values), 2),
        'median_rating': round(statistics.median(rating_values), 2),
        'std_dev_rating': round(statistics.stdev(rating_values) if len(rating_values) > 1 else 0, 2),
        'min_rating': min(rating_values),
        'max_rating': max(rating_values),
        'total_matches': sum(match_counts),
        'avg_matches_per_question': round(statistics.mean(match_counts), 2),
        'questions_with_matches': sum(1 for m in match_counts if m > 0),
        'last_updated': datetime.now().isoformat()
    }
    return stats

def save_ratings(ratings):
    """
    Save Elo ratings to the JSON file with statistics.
    """
    stats = calculate_stats(ratings)
    output = {
        'statistics': stats,
        'ratings': ratings
    }
    with open(RATINGS_FILE, 'w') as f:
        json.dump(output, f, indent=2, sort_keys=True)

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
    parser.add_argument('--batch-size', '-b', type=int, default=8,
                        help='Number of questions to rank per iteration')
    parser.add_argument('--num-questions-per-iteration', '-q', type=int,
                        dest='batch_size',
                        help='Alias for --batch-size')
    parser.add_argument('--iterations', '-n', type=int, default=10,
                        help='Number of ranking iterations')
    parser.add_argument('--model', '-m', type=str, default='gpt-4o',
                        help='LLM model to use for ranking')
    parser.add_argument('--k-factor', '-k', type=float, default=128,
                        help='Total K-factor per batch (controls volatility of rating changes; per-pair K = k_factor/(n-1))')
    parser.add_argument('--sleep', '-s', type=float, default=0,
                        help='Seconds to sleep between iterations')
    parser.add_argument('--concurrent-batches', '-c', type=int, default=40,
                        help='Number of batches to process concurrently')
    args = parser.parse_args()
    # Load or init ratings
    ratings = load_ratings()
    all_ids = list(ratings.keys())
    
    def weighted_sample(question_ids, ratings, batch_size):
        """
        Sample questions with higher probability for those with fewer matches.
        """
        if len(question_ids) <= batch_size:
            return question_ids[:]
        
        # Calculate weights: higher weight for fewer matches
        max_matches = max(ratings[qid]['matches'] for qid in question_ids)
        weights = []
        for qid in question_ids:
            matches = ratings[qid]['matches']
            # Weight inversely proportional to matches + 1 (to avoid division by zero)
            weight = 1.0 / (matches + 1)
            weights.append(weight)
        
        # Sample with replacement based on weights
        return random.choices(question_ids, weights=weights, k=batch_size)
    
    # Print initial statistics
    stats = calculate_stats(ratings)
    print(f"\n=== Initial Statistics ===")
    print(f"Total questions: {stats.get('total_questions', 0)}")
    print(f"Average rating: {stats.get('average_rating', 0)}")
    print(f"Rating std dev: {stats.get('std_dev_rating', 0)}")
    print(f"Questions with matches: {stats.get('questions_with_matches', 0)}")
    print(f"Total matches: {stats.get('total_matches', 0)}")
    print(f"\n=== Configuration ===")
    print(f"Batch size: {args.batch_size}")
    print(f"Concurrent batches: {args.concurrent_batches}")
    print(f"Iterations: {args.iterations}")
    print(f"Model: {args.model}")
    print(f"K-factor: {args.k_factor}")
    print("="*50)
    
    async def process_concurrent_batches():
        nonlocal ratings
        for i in range(args.iterations):
            print(f"\n[Iteration {i+1}/{args.iterations}] Creating {args.concurrent_batches} batches...")
            
            # Create multiple batches for concurrent processing using weighted sampling
            batches = []
            for batch_idx in range(args.concurrent_batches):
                batch = weighted_sample(all_ids, ratings, args.batch_size)
                batches.append(batch)
                avg_matches = sum(ratings[qid]['matches'] for qid in batch) / len(batch)
                print(f"  Batch {batch_idx+1}: {len(batch)} questions (avg matches: {avg_matches:.1f})")
            
            print(f"[Iteration {i+1}/{args.iterations}] Starting concurrent ranking...")
            start_time = time.time()
            
            # Process batches concurrently
            tasks = [rank_questions(batch, model=args.model) for batch in batches]
            rankings = await asyncio.gather(*tasks, return_exceptions=True)
            
            ranking_time = time.time() - start_time
            print(f"[Iteration {i+1}/{args.iterations}] Ranking completed in {ranking_time:.2f}s")
            
            # Update ratings for successful rankings
            successful_batches = 0
            failed_batches = 0
            total_questions_processed = 0
            
            for batch_idx, (batch, ranking) in enumerate(zip(batches, rankings)):
                if isinstance(ranking, Exception) or not ranking:
                    print(f"  Batch {batch_idx+1}: FAILED - {str(ranking)[:100]}")
                    failed_batches += 1
                    continue
                ratings = update_ratings(ratings, batch, ranking, k_factor=args.k_factor)
                successful_batches += 1
                total_questions_processed += len(set(batch))  # unique questions
                print(f"  Batch {batch_idx+1}: SUCCESS - {len(set(batch))} unique questions")
            
            if successful_batches > 0:
                save_ratings(ratings)
                new_stats = calculate_stats(ratings)
                print(f"\n[Iteration {i+1}/{args.iterations}] RESULTS:")
                print(f"  ✓ Successful: {successful_batches}/{args.concurrent_batches}")
                print(f"  ✗ Failed: {failed_batches}/{args.concurrent_batches}")
                print(f"  Questions processed: {total_questions_processed}")
                print(f"  New avg rating: {new_stats.get('average_rating', 0)}")
                print(f"  New std dev: {new_stats.get('std_dev_rating', 0)}")
                print(f"  Total matches: {new_stats.get('total_matches', 0)}")
            else:
                print(f"\n[Iteration {i+1}/{args.iterations}] ✗ FAILED - no successful batches")
            
            if args.sleep and i < args.iterations - 1:
                print(f"Sleeping {args.sleep}s...")
                time.sleep(args.sleep)
    
    asyncio.run(process_concurrent_batches())

if __name__ == '__main__':  # pragma: no cover
    main()