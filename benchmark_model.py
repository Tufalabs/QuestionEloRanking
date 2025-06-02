#!/usr/bin/env python3
"""
Script to benchmark a model's ELO rating by testing it against sampled questions.
Uses LLM as a judge to verify answers and calculate accuracy-based ELO.
"""

import json
import argparse
import sys
import re
import random
import numpy as np
from pathlib import Path
import openai
import os
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import time
from typing import List, Dict, Tuple

def load_json(filename):
    """Load JSON data from file."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {filename} not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {filename}")
        sys.exit(1)

def sample_questions_by_distribution(questions_data, elos_data, num_questions, distribution='uniform'):
    """Sample questions according to specified distribution."""
    # Get all questions with ELO ratings
    questions_with_elo = []
    questions_by_id = {q['id']: q for q in questions_data['questions']}
    
    for question_id, elo_info in elos_data['ratings'].items():
        if question_id in questions_by_id:
            question = questions_by_id[question_id].copy()
            question['elo_rating'] = elo_info['rating']
            questions_with_elo.append(question)
    
    if len(questions_with_elo) < num_questions:
        print(f"Warning: Only {len(questions_with_elo)} questions available, using all")
        return questions_with_elo
    
    if distribution == 'uniform':
        return random.sample(questions_with_elo, num_questions)
    elif distribution == 'normal':
        # Sample from normal distribution around median ELO
        elos = [q['elo_rating'] for q in questions_with_elo]
        median_elo = np.median(elos)
        std_elo = np.std(elos)
        
        # Generate target ELOs from normal distribution
        target_elos = np.random.normal(median_elo, std_elo/2, num_questions)
        target_elos = np.clip(target_elos, min(elos), max(elos))
        
        # Find closest questions to target ELOs
        selected = []
        used_indices = set()
        
        for target_elo in target_elos:
            # Find closest unused question
            best_idx = None
            best_diff = float('inf')
            
            for i, q in enumerate(questions_with_elo):
                if i in used_indices:
                    continue
                diff = abs(q['elo_rating'] - target_elo)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i
            
            if best_idx is not None:
                selected.append(questions_with_elo[best_idx])
                used_indices.add(best_idx)
        
        return selected
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

async def query_model_async(question: str, model_name: str, api_key: str, semaphore: asyncio.Semaphore) -> str:
    """Query the model to get an answer asynchronously."""
    async with semaphore:
        client = openai.AsyncOpenAI(api_key=api_key)
        
        prompt = f"""Please solve this math problem step by step and provide your final answer in <answer></answer> tags.

Question: {question}

Think through this carefully and show your work. Put only your final numerical answer or simplified expression in the answer tags."""
        
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error querying model: {e}")
            return ""

def query_model(question: str, model_name: str, api_key: str) -> str:
    """Query the model to get an answer (sync version for backward compatibility)."""
    client = openai.OpenAI(api_key=api_key)
    
    prompt = f"""Please solve this math problem step by step and provide your final answer in <answer></answer> tags.

Question: {question}

Think through this carefully and show your work. Put only your final numerical answer or simplified expression in the answer tags."""
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error querying model: {e}")
        return ""

def extract_answer(response: str) -> str:
    """Extract answer from model response."""
    if not response:
        return ""
    
    # Look for answer tags first
    match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        # Clean up the answer - remove extra text, keep only the numerical part
        cleaned = clean_numerical_answer(answer)
        return cleaned if cleaned else answer
    
    # Fallback: look for patterns like "The answer is X" or "= X" at the end
    lines = response.strip().split('\n')
    
    # Check last few lines for answer patterns
    for line in reversed(lines[-5:]):
        line = line.strip()
        if not line:
            continue
            
        # Look for "answer is", "= ", "answer: " patterns
        patterns = [
            r'(?:answer is|answer:|result is|result:|equals?)\s*([+-]?\d+(?:\.\d+)?(?:/\d+)?)',
            r'=\s*([+-]?\d+(?:\.\d+)?(?:/\d+)?)\s*$',
            r'^([+-]?\d+(?:\.\d+)?(?:/\d+)?)\s*$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return match.group(1).strip()
    
    # Last resort: find any number in the response
    numbers = re.findall(r'[+-]?\d+(?:\.\d+)?(?:/\d+)?', response)
    if numbers:
        return numbers[-1]  # Return the last number found
    
    return ""

def clean_numerical_answer(answer: str) -> str:
    """Clean numerical answer to extract just the number."""
    if not answer:
        return ""
    
    # Remove common non-numerical text
    answer = re.sub(r'(?:the answer is|answer:|result is|result:|equals?|approximately|about)\s*', '', answer, flags=re.IGNORECASE)
    
    # Extract number patterns
    match = re.search(r'([+-]?\d+(?:\.\d+)?(?:/\d+)?)', answer.strip())
    if match:
        return match.group(1)
    
    return answer.strip()

async def judge_answer_async(question: str, correct_answer: str, model_answer: str, judge_model: str, api_key: str, semaphore: asyncio.Semaphore, debug: bool = False) -> bool:
    """Use LLM judge to verify if the answer is correct asynchronously."""
    # Return False immediately for empty or None answers
    if not model_answer or model_answer.strip() == "":
        if debug:
            print(f"  DEBUG: Empty model answer -> INCORRECT")
        return False
    
    async with semaphore:
        client = openai.AsyncOpenAI(api_key=api_key)
        
        prompt = f"""You are a strict math teacher grading student answers. Compare the student's final answer to the correct answer.

Question: {question}

CORRECT ANSWER: {correct_answer}
STUDENT ANSWER: {model_answer}

Rules for grading:
1. Only mark CORRECT if the student's numerical answer matches the correct answer exactly
2. Consider equivalent forms: 0.5 = 1/2, 4 = 4.0, 2/4 = 1/2
3. Ignore extra text or explanation - focus only on the final numerical value
4. If the student answer is empty, unclear, or wrong, mark INCORRECT
5. If the student gives multiple different answers, mark INCORRECT
6. Be very strict - when in doubt, mark INCORRECT

Examples:
- Correct: 42, Student: 42 → CORRECT
- Correct: 1/2, Student: 0.5 → CORRECT  
- Correct: 3, Student: The answer is 3 → CORRECT
- Correct: 5, Student: 4 → INCORRECT
- Correct: 2, Student: I think it's 2 or maybe 3 → INCORRECT
- Correct: 10, Student: → INCORRECT (empty)

Respond with exactly one word: "CORRECT" or "INCORRECT" """
        
        try:
            response = await client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10
            )
            result = response.choices[0].message.content.strip().upper()
            is_correct = result == "CORRECT"
            
            if debug:
                print(f"  DEBUG: Correct='{correct_answer}', Model='{model_answer}', Judge='{result}' -> {is_correct}")
            
            return is_correct
        except Exception as e:
            print(f"Error with judge model: {e}")
            return False

def judge_answer(question: str, correct_answer: str, model_answer: str, judge_model: str, api_key: str) -> bool:
    """Use LLM judge to verify if the answer is correct (sync version for backward compatibility)."""
    client = openai.OpenAI(api_key=api_key)
    
    prompt = f"""You are a math teacher evaluating student answers. Determine if the student's answer is mathematically equivalent to the correct answer.

Question: {question}

Correct Answer: {correct_answer}
Student Answer: {model_answer}

Consider these as equivalent:
- Different forms of the same number (e.g., 0.5 = 1/2 = 50%)
- Simplified vs unsimplified fractions
- Decimal vs fraction representations
- Different valid mathematical expressions for the same value

Respond with only "CORRECT" or "INCORRECT"."""
    
    try:
        response = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=50
        )
        result = response.choices[0].message.content.strip().upper()
        return "CORRECT" in result
    except Exception as e:
        print(f"Error with judge model: {e}")
        return False

def calculate_elo_from_accuracy(accuracy: float, question_elos: List[float]) -> float:
    """Calculate model ELO based on accuracy and question difficulties."""
    if not question_elos:
        return 1000  # Default ELO
    
    avg_question_elo = np.mean(question_elos)
    
    # Convert accuracy to ELO using logistic function
    # accuracy = 1 / (1 + 10^((opponent_elo - player_elo) / 400))
    # Solving for player_elo when opponent_elo is avg_question_elo
    
    if accuracy >= 0.99:
        accuracy = 0.99  # Avoid log(0)
    elif accuracy <= 0.01:
        accuracy = 0.01
    
    # player_elo = opponent_elo + 400 * log10((1 - accuracy) / accuracy)
    model_elo = avg_question_elo + 400 * np.log10((1 - accuracy) / accuracy)
    
    return round(model_elo)

def save_results(model_name: str, results: Dict, output_dir: str):
    """Save benchmark results to JSON file."""
    output_file = f"{output_dir}/benchmark_{model_name.replace('/', '_')}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")

async def process_question_async(question: Dict, model_name: str, judge_model: str, api_key: str, semaphore: asyncio.Semaphore, question_num: int, total_questions: int, debug: bool = False) -> Dict:
    """Process a single question asynchronously."""
    if question_num <= 10 or debug:  # Show debug for first 10 questions
        print(f"Testing question {question_num}/{total_questions}...")
    
    # Get model response
    model_response = await query_model_async(question['question'], model_name, api_key, semaphore)
    model_answer = extract_answer(model_response)
    
    # Judge the answer
    is_correct = await judge_answer_async(
        question['question'], 
        question['answer'], 
        model_answer, 
        judge_model, 
        api_key,
        semaphore,
        debug=(question_num <= 10 or debug)  # Debug first 10 questions
    )
    
    if question_num <= 10 or debug:
        print(f"  Q{question_num}: Expected='{question['answer']}', Got='{model_answer}', Correct={is_correct}")
    elif question_num % 50 == 0:
        print(f"  Processed {question_num}/{total_questions} questions...")
    
    return {
        "question_id": question['id'],
        "question": question['question'],
        "correct_answer": question['answer'],
        "model_response": model_response,
        "extracted_answer": model_answer,
        "is_correct": is_correct,
        "question_elo": question['elo_rating']
    }

async def benchmark_concurrent(sampled_questions: List[Dict], model_name: str, judge_model: str, api_key: str, max_concurrent: int, debug: bool = False) -> Tuple[List[Dict], int]:
    """Run benchmark with concurrent processing."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    tasks = []
    for i, question in enumerate(sampled_questions):
        task = process_question_async(question, model_name, judge_model, api_key, semaphore, i+1, len(sampled_questions), debug)
        tasks.append(task)
    
    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)
    
    # Count correct answers
    correct_count = sum(1 for result in results if result['is_correct'])
    
    return results, correct_count

def main():
    parser = argparse.ArgumentParser(description='Benchmark model ELO rating')
    parser.add_argument('model_name', help='Model name to benchmark (e.g., gpt-4, gpt-3.5-turbo)')
    parser.add_argument('--num_questions', '-n', type=int, default=50, 
                       help='Number of questions to sample (default: 50)')
    parser.add_argument('--distribution', '-d', choices=['uniform', 'normal'], default='uniform',
                       help='Sampling distribution (default: uniform)')
    parser.add_argument('--judge_model', '-j', default='gpt-4', 
                       help='Model to use as judge (default: gpt-4)')
    parser.add_argument('--api_key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--output_dir', '-o', default='json', 
                       help='Output directory (default: json)')
    parser.add_argument('--max_concurrent', '-c', type=int, default=100,
                       help='Maximum concurrent requests (default: 10)')
    parser.add_argument('--use_concurrent', action='store_true',
                       help='Use concurrent processing (faster but may hit rate limits)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output to see answer comparisons')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OpenAI API key required. Set OPENAI_API_KEY env var or use --api_key")
        sys.exit(1)
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    print(f"Loading data...")
    questions_data = load_json('questions.json')
    elos_data = load_json('question_elos.json')
    
    print(f"Sampling {args.num_questions} questions using {args.distribution} distribution...")
    sampled_questions = sample_questions_by_distribution(
        questions_data, elos_data, args.num_questions, args.distribution
    )
    
    print(f"Benchmarking model {args.model_name}...")
    results = {
        "model_name": args.model_name,
        "judge_model": args.judge_model,
        "num_questions": len(sampled_questions),
        "distribution": args.distribution,
        "max_concurrent": args.max_concurrent if args.use_concurrent else 1,
        "use_concurrent": args.use_concurrent,
        "questions_tested": [],
        "correct_answers": 0,
        "total_questions": len(sampled_questions),
        "accuracy": 0.0,
        "estimated_elo": 0,
        "avg_question_elo": 0.0
    }
    
    question_elos = [q['elo_rating'] for q in sampled_questions]
    
    start_time = time.time()
    
    if args.use_concurrent:
        print(f"Running with {args.max_concurrent} concurrent requests...")
        # Use concurrent processing
        question_results, correct_count = asyncio.run(
            benchmark_concurrent(sampled_questions, args.model_name, args.judge_model, api_key, args.max_concurrent, args.debug)
        )
        results["questions_tested"] = question_results
    else:
        print("Running sequentially...")
        # Use sequential processing
        correct_count = 0
        for i, question in enumerate(sampled_questions):
            print(f"Testing question {i+1}/{len(sampled_questions)}...")
            
            # Get model response
            model_response = query_model(question['question'], args.model_name, api_key)
            model_answer = extract_answer(model_response)
            
            # Judge the answer
            is_correct = judge_answer(
                question['question'], 
                question['answer'], 
                model_answer, 
                args.judge_model, 
                api_key
            )
            
            if is_correct:
                correct_count += 1
            
            # Store detailed results
            results["questions_tested"].append({
                "question_id": question['id'],
                "question": question['question'],
                "correct_answer": question['answer'],
                "model_response": model_response,
                "extracted_answer": model_answer,
                "is_correct": is_correct,
                "question_elo": question['elo_rating']
            })
            
            print(f"  Question ELO: {question['elo_rating']}, Correct: {is_correct}")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Calculate final results
    accuracy = correct_count / len(sampled_questions)
    estimated_elo = calculate_elo_from_accuracy(accuracy, question_elos)
    avg_question_elo = np.mean(question_elos)
    
    results.update({
        "correct_answers": correct_count,
        "accuracy": round(accuracy, 4),
        "estimated_elo": estimated_elo,
        "avg_question_elo": round(avg_question_elo, 1),
        "processing_time_seconds": round(processing_time, 2)
    })
    
    # Print summary
    print(f"\n=== Benchmark Results ===")
    print(f"Model: {args.model_name}")
    print(f"Questions tested: {len(sampled_questions)}")
    print(f"Correct answers: {correct_count}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Average question ELO: {avg_question_elo:.1f}")
    print(f"Estimated model ELO: {estimated_elo}")
    print(f"Processing time: {processing_time:.2f} seconds")
    if args.use_concurrent:
        print(f"Concurrent requests: {args.max_concurrent}")
    
    # Save results
    save_results(args.model_name, results, args.output_dir)

if __name__ == "__main__":
    main()