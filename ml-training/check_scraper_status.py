#!/usr/bin/env python3

import json
import os
import glob

def check_scraper_status():
    """Check status and failures of all active scraper sessions"""
    
    # Find all scraper state files
    state_files = glob.glob("data/structured_collection_*/scraper_state.json")
    
    if not state_files:
        print("No active scraper sessions found.")
        return
        
    for state_file in state_files:
        session_name = state_file.split("/")[1].replace("structured_collection_", "")
        print(f"\n🔍 SESSION: {session_name}")
        print(f"📁 State file: {state_file}")
        
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            # Basic stats
            print(f"📊 STATS:")
            print(f"  Total games processed: {state.get('total_games_processed', 0)}")
            print(f"  Total images collected: {state.get('total_images_collected', 0)}")
            print(f"  Consecutive failures: {state.get('consecutive_failures', 0)}")
            print(f"  Total failed games: {len(state.get('failed_games', []))}")
            
            # Current position
            print(f"📍 POSITION:")
            print(f"  Week: {state.get('current_week', 'Unknown')}")
            print(f"  Match: {state.get('current_match_in_week', 'Unknown')}")
            print(f"  Round: {state.get('current_round', 'Unknown')}")
            print(f"  Game: {state.get('current_game', 'Unknown')}")
            
            # Last successful game
            if state.get('last_successful_game'):
                last_success = state['last_successful_game']
                print(f"✅ LAST SUCCESS:")
                print(f"  Game: {last_success['url']}")
                print(f"  Time: {last_success['timestamp']}")
            
            # Recent failures
            failed_games = state.get('failed_games', [])
            if failed_games:
                print(f"❌ RECENT FAILURES:")
                for game in failed_games[-3:]:  # Show last 3 failures
                    print(f"  - W{game['week']} M{game['match']} R{game['round']} G{game['game']}: {game['url']}")
                    print(f"    Time: {game['timestamp']}")
            
        except Exception as e:
            print(f"❌ Error reading state file: {e}")
            
    # Check for running processes
    print(f"\n🔄 RUNNING PROCESSES:")
    os.system("ps aux | grep -v grep | grep structured_scraper || echo 'No scrapers running'")

if __name__ == "__main__":
    check_scraper_status()