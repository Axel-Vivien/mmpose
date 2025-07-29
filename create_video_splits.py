#!/usr/bin/env python3
"""
Script to split 180,000 videos from action_timespans.json into 8 equal batches
maintaining the same category distribution (true_positive, false_positive, unbiased)
"""

import json
import os
from collections import defaultdict
import random

def load_action_timespans(filepath):
    """Load the action_timespans.json file"""
    print(f"Loading {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} videos")
    return data

def categorize_videos(data):
    """Categorize videos by their path (true_positive, false_positive, unbiased)"""
    categories = defaultdict(dict)
    
    for video_path, video_data in data.items():
        # Extract category from path: data-sample_2025-06-02/videos/2025-XX-XX/CATEGORY/filename.mp4
        path_parts = video_path.split('/')
        if len(path_parts) >= 4:
            category = path_parts[3]  # Index 3 should be the category
            categories[category][video_path] = video_data
    
    return categories

def create_batches(categories, num_batches=8):
    """Split each category into equal batches"""
    all_batches = []
    
    # Calculate videos per batch for each category
    videos_per_batch = {}
    for category, videos in categories.items():
        total_videos = len(videos)
        per_batch = total_videos // num_batches
        videos_per_batch[category] = per_batch
        print(f"{category}: {total_videos} total, {per_batch} per batch")
    
    # Create 8 empty batches
    for i in range(num_batches):
        all_batches.append({})
    
    # Distribute videos from each category across batches
    for category, videos in categories.items():
        video_items = list(videos.items())
        random.shuffle(video_items)  # Randomize to ensure good distribution
        
        videos_per_batch_count = videos_per_batch[category]
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * videos_per_batch_count
            end_idx = start_idx + videos_per_batch_count
            
            # Handle any remainder in the last batch
            if batch_idx == num_batches - 1:
                end_idx = len(video_items)
            
            batch_videos = video_items[start_idx:end_idx]
            
            for video_path, video_data in batch_videos:
                all_batches[batch_idx][video_path] = video_data
    
    return all_batches

def save_batch_files(batches, output_dir=".", prefix="video_batch"):
    """Save each batch to a separate JSON file"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    batch_filenames = []
    
    for i, batch in enumerate(batches, 1):
        filename = f"{prefix}_{i:02d}_of_{len(batches):02d}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(batch, f, indent=2)
        
        batch_filenames.append(filename)
        print(f"Saved {filename} with {len(batch)} videos")
    
    return batch_filenames

def create_summary(batches, batch_filenames, original_filename="action_timespans.json"):
    """Create a summary file similar to split_summary.json"""
    
    # Calculate total videos and distribution
    total_videos = sum(len(batch) for batch in batches)
    videos_per_batch = [len(batch) for batch in batches]
    
    # Get sample videos from each batch
    batch_details = []
    for i, (batch, filename) in enumerate(zip(batches, batch_filenames), 1):
        sample_videos = list(batch.keys())[:3]  # Take first 3 as samples
        
        batch_details.append({
            "batch_number": i,
            "filename": filename,
            "video_count": len(batch),
            "sample_videos": sample_videos
        })
    
    summary = {
        "split_summary": {
            "original_file": original_filename,
            "total_videos": total_videos,
            "number_of_batches": len(batches),
            "videos_per_batch": videos_per_batch,
            "batch_files": batch_filenames
        },
        "batch_details": batch_details
    }
    
    return summary

def save_summary(summary, output_dir=".", filename="batch_summary.json"):
    """Save the summary to a JSON file"""
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {filename}")

def print_distribution_analysis(categories, batches):
    """Print analysis of the distribution"""
    print("\n=== DISTRIBUTION ANALYSIS ===")
    
    # Original distribution
    print("\nOriginal distribution:")
    total_original = sum(len(videos) for videos in categories.values())
    for category, videos in categories.items():
        count = len(videos)
        percentage = (count / total_original) * 100
        print(f"  {category}: {count:,} videos ({percentage:.1f}%)")
    
    # Batch distribution
    print(f"\nBatch distribution ({len(batches)} batches):")
    for i, batch in enumerate(batches, 1):
        batch_categories = defaultdict(int)
        for video_path in batch.keys():
            category = video_path.split('/')[3]
            batch_categories[category] += 1
        
        total_batch = len(batch)
        print(f"  Batch {i}: {total_batch:,} videos")
        for category, count in batch_categories.items():
            percentage = (count / total_batch) * 100
            print(f"    {category}: {count:,} ({percentage:.1f}%)")

def main():
    # Set random seed for reproducible results
    random.seed(42)
    
    # Configuration
    input_file = "action_timespans.json"
    output_dir = "video_batches"
    num_batches = 8
    
    # Load and process data
    data = load_action_timespans(input_file)
    categories = categorize_videos(data)
    
    print(f"\nFound categories: {list(categories.keys())}")
    for category, videos in categories.items():
        print(f"  {category}: {len(videos):,} videos")
    
    # Create batches
    print(f"\nCreating {num_batches} batches...")
    batches = create_batches(categories, num_batches)
    
    # Save batch files
    print(f"\nSaving batch files to {output_dir}/...")
    batch_filenames = save_batch_files(batches, output_dir)
    
    # Create and save summary
    summary = create_summary(batches, batch_filenames, input_file)
    save_summary(summary, output_dir)
    
    # Print analysis
    print_distribution_analysis(categories, batches)
    
    print(f"\n=== COMPLETED ===")
    print(f"Created {len(batches)} batch files in '{output_dir}/' directory")
    print(f"Each batch contains ~{len(batches[0]):,} videos")
    print(f"Summary saved as 'batch_summary.json'")

if __name__ == "__main__":
    main() 