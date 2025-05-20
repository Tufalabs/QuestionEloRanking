# Question Elo Ranking

A command-line tool to maintain Elo-style difficulty ratings for a pool of math questions by leveraging an LLM to rank small batches of questions from easiest to hardest.

## Overview

This script picks a random batch of questions from `merged_questions.json`, prompts your chosen language model to rank them in order of increasing difficulty, and then updates each question’s Elo rating based on the new ranking. Ratings are stored in `question_elos.json`.

## Prerequisites
- Python 3.7 or higher
- An API key for your chosen LLM provider (e.g., OpenAI, Anthropic, DeepInfra, etc.)
- The following Python packages:
  - openai
  - anthropic
  - aiohttp
  - requests

You can install the required packages via pip:
```bash
pip install openai anthropic aiohttp requests
```

## Configuration
1. Clone this repository and `cd` into the project root.
2. Set your environment variables. For example, to use OpenAI:
   ```bash
   export OPENAI_API_KEY="<your-openai-api-key>"
   ```
   Or source the provided `set_api_keys.sh` if you have multiple keys:
   ```bash
   source set_api_keys.sh
   ```

## Usage
```bash
python utils/elo.py [OPTIONS]
```

### Options
- `-b, --batch-size INT`             Number of questions to rank per iteration (default: 20)
- `-q, --num-questions-per-iteration INT`
                                      Alias for `--batch-size`
- `-n, --iterations INT`             Number of iterations to run (default: 1)
- `-m, --model TEXT`                 LLM model name (e.g., `gpt-4o-mini`, `claude-2`) (default: `gpt-4o-mini`)
- `-k, --k-factor FLOAT`             Total K-factor per batch; higher → more volatile rating updates (default: 32)
- `-s, --sleep FLOAT`                Seconds to sleep between iterations (default: 0)
- `-h, --help`                       Show help message and exit

### Example
Rank 30 questions per iteration over 5 iterations using GPT-4-mini with a K-factor of 16:
```bash
python utils/elo.py -b 30 -n 5 -k 16
```

## How It Works
- **Batch selection**: randomly sample a subset of question IDs
- **LLM ranking**: prompt the model to return a JSON array of IDs sorted easiest→hardest
- **Rating update**: for each batch, compute pairwise expected scores and actual scores (based on rank), then adjust ratings by `(S - E) * (k_factor/(n-1))` per question

## Files
- `merged_questions.json`: source question bank (id, text, source)
- `question_elos.json`: stores current Elo data per question (updated after each run), as JSON mapping question_id to:
  - `rating` (int): current Elo rating
  - `matches` (int): cumulative number of pairwise comparisons the question has participated in
  - `uncertainty` (float|null): ratio 1/sqrt(matches), an approximate confidence measure (null if no comparisons yet)
  - `uncertainty_points` (float|null): uncertainty expressed in Elo points (≈400×uncertainty ratio)
- `utils/elo.py`: main script implementing the Elo workflow
- `utils/inference.py`: wrapper for async LLM calls

## K-Factor
The K-factor determines how much ratings can change in each batch. A higher K-factor yields faster, larger adjustments; a lower value makes ratings more stable. Internally, the script divides the total K-factor by `(batch_size - 1)` to scale each pairwise comparison.
