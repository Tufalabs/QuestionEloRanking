#!/usr/bin/env python3
"""
Script to sample questions within a specified ELO rating range.
Reads from question_elos.json and questions.json, filters by ELO range,
and saves the result to filtered_elo.json.
"""

import json
import argparse
import sys

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

def filter_questions_by_elo(questions_data, elos_data, min_elo, max_elo):
    """Filter questions by ELO rating range."""
    filtered_questions = []
    
    # Create a lookup dict for questions by ID
    questions_by_id = {q['id']: q for q in questions_data['questions']}
    
    # Filter questions based on ELO range
    for question_id, elo_info in elos_data['ratings'].items():
        rating = elo_info['rating']
        
        if min_elo <= rating <= max_elo:
            if question_id in questions_by_id:
                # Add ELO info to the question
                question = questions_by_id[question_id].copy()
                question['elo_rating'] = rating
                question['elo_matches'] = elo_info['matches']
                question['elo_uncertainty'] = elo_info['uncertainty']
                question['elo_uncertainty_points'] = elo_info['uncertainty_points']
                filtered_questions.append(question)
    
    return filtered_questions

def save_filtered_questions(filtered_questions, output_file):
    """Save filtered questions to JSON file."""
    output_data = {
        "total_questions": len(filtered_questions),
        "elo_filtered": True,
        "questions": filtered_questions
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Saved {len(filtered_questions)} questions to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Filter questions by ELO rating range')
    parser.add_argument('min_elo', type=int, help='Minimum ELO rating')
    parser.add_argument('max_elo', type=int, help='Maximum ELO rating')
    parser.add_argument('--output', '-o', default='json/filtered_elo.json', 
                       help='Output filename (default: json/filtered_elo.json)')
    
    args = parser.parse_args()
    
    if args.min_elo > args.max_elo:
        print("Error: min_elo must be less than or equal to max_elo")
        sys.exit(1)
    
    print(f"Loading data...")
    questions_data = load_json('questions.json')
    elos_data = load_json('question_elos.json')
    
    print(f"Filtering questions with ELO between {args.min_elo} and {args.max_elo}...")
    filtered_questions = filter_questions_by_elo(questions_data, elos_data, args.min_elo, args.max_elo)
    
    if not filtered_questions:
        print(f"No questions found in ELO range {args.min_elo}-{args.max_elo}")
        sys.exit(1)
    
    save_filtered_questions(filtered_questions, args.output)

if __name__ == "__main__":
    main()