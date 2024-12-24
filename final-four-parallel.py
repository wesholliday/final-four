#!/usr/bin/env python
# coding: utf-8

"""
This notebook is a parallelized version of final-four.ipynb, supplementary material for the paper 
"A simple Condorcet voting method for Final Four elections" by Wesley H. Holliday.
"""

import pref_voting
from pref_voting.profiles import *
from pref_voting.voting_methods import *
from pref_voting.combined_methods import *
from pref_voting.margin_based_methods import *
from pref_voting.generate_profiles import generate_profile 
from pref_voting.generate_weighted_majority_graphs import generate_edge_ordered_tournament_infinite_limit
from pref_voting.profiles_with_ties import ProfileWithTies
from pref_voting.io.readers import preflib_to_profile

from tqdm import tqdm
import networkx as nx
import numpy as np
import glob
from zipfile import ZipFile
import io
import itertools
from itertools import combinations
from preflibtools.instances import OrdinalInstance

# Print version for reference
print(pref_voting.__version__)

# # 1. Figure 1 percentages
# First we create networkx digraphs for the four tournament isomorphism types.

# Create the tournament types
linear_order = nx.DiGraph()
linear_order.add_nodes_from(range(4))
linear_order.add_edges_from([(0,1),(0,2), (0,3), (1,2), (1,3), (2,3)]) 

bottom_cycle = nx.DiGraph()
bottom_cycle.add_nodes_from(range(4))
bottom_cycle.add_edges_from([(0,1),(0,2), (0,3), (1,2), (2,3), (3,1)])

top_cycle = nx.DiGraph()
top_cycle.add_nodes_from(range(4))
top_cycle.add_edges_from([(0,1),(1,2), (2,0), (0,3), (1,3), (2,3)])

four_cycle = nx.DiGraph()
four_cycle.add_nodes_from(range(4))
four_cycle.add_edges_from([(0,1),(1,2), (2,3), (3,0), (2,0), (1,3)])

# Define the proposed method
@vm(name = "Proposed Method")
def proposed_method(edata, curr_cands = None):
    copeland_winners = copeland(edata)
    worst_loss = dict()
    
    for c in copeland_winners:
        worst_loss[c] = max([edata.margin(c2, c) for c2 in edata.candidates])
    
    smallest_worst_loss = min(worst_loss.values())
    copeland_winners_with_smallest_worst_loss = [c for c in copeland_winners if worst_loss[c] == smallest_worst_loss]
    
    return copeland_winners_with_smallest_worst_loss

# Section 1: Tournament statistics loop (Parallelized)
from pathos.multiprocessing import ProcessingPool as Pool
import os

def process_single_tournament(_):
    """Process a single tournament and classify its type."""
    mg = generate_edge_ordered_tournament_infinite_limit(4)
    
    # Extract the underlying directed graph from mg
    g = nx.DiGraph()
    g.add_nodes_from(range(4))
    directed_edges = [(a,b) for (a,b,c) in mg.edges]
    g.add_edges_from(directed_edges)
    
    result = {
        'linear_order': 0,
        'bottom_cycle': 0,
        'ascending_top_cycle': 0,
        'descending_top_cycle': 0,
        'SL_four_cycle': 0,
        'LS_four_cycle': 0
    }
    
    # Find the appropriate isomorphism type
    if nx.is_isomorphic(g, linear_order):
        result['linear_order'] = 1
    
    elif nx.is_isomorphic(g, bottom_cycle):
        result['bottom_cycle'] = 1
    
    elif nx.is_isomorphic(g, top_cycle):
        top_cycle_cands = copeland(mg)
        assert len(top_cycle_cands) == 3
        
        # Find the candidate in the top cycle with the largest win
        max_margin_in_top_cycle = max([mg.margin(c1, c2) for c1 in top_cycle_cands for c2 in top_cycle_cands if c1 != c2])
        top_cycle_cands_with_largest_win = [c for c in top_cycle_cands if max([mg.margin(c, c2) for c2 in top_cycle_cands]) == max_margin_in_top_cycle]
        assert len(top_cycle_cands_with_largest_win) == 1
        top_cycle_cand_with_largest_win = top_cycle_cands_with_largest_win[0]
        
        # Find the candidate in the top cycle with the smallest loss
        min_pos_margin_in_top_cycle = min([mg.margin(c1, c2) for c1 in top_cycle_cands for c2 in top_cycle_cands if mg.margin(c1,c2) > 0])
        top_cycle_cands_with_smallest_loss = [c for c in top_cycle_cands if min([mg.margin(c2, c) for c2 in top_cycle_cands if mg.margin(c2,c) > 0]) == min_pos_margin_in_top_cycle]
        assert len(top_cycle_cands_with_smallest_loss) == 1
        top_cycle_cand_with_smallest_loss = top_cycle_cands_with_smallest_loss[0]
        
        if top_cycle_cand_with_smallest_loss != top_cycle_cand_with_largest_win:
            result['ascending_top_cycle'] = 1
        else:
            result['descending_top_cycle'] = 1
    
    elif nx.is_isomorphic(g, four_cycle):
        copeland_winners = copeland(mg)
        assert len(copeland_winners) == 2
        margin_between_copeland_winners = max([mg.margin(c1, c2) for c1 in copeland_winners for c2 in copeland_winners]) 
        margin_of_non_copeland_over_copeland = max([mg.margin(c, c2) for c in mg.candidates if c not in copeland_winners for c2 in copeland_winners])
        
        if margin_of_non_copeland_over_copeland < margin_between_copeland_winners:
            result['SL_four_cycle'] = 1
        else:
            result['LS_four_cycle'] = 1
            
    return result

def compare_methods(_):
    """Compare proposed method with other voting methods for a single tournament."""
    mg = generate_edge_ordered_tournament_infinite_limit(4)
    
    pm = proposed_method(mg)
    rp = ranked_pairs(mg)
    mm = minimax(mg)
    smm = smith_minimax(mg)
    bp = beat_path(mg)
    sv = stable_voting(mg)
    rv = river(mg)
    
    return {
        'rp': int(pm != rp),
        'mm': int(pm != mm),
        'smm': int(pm != smm),
        'bp': int(pm != bp),
        'sv': int(pm != sv),
        'rv': int(pm != rv)
    }

def check_heavy_top_cycle(_):
    """Check for heavy top cycle in a single tournament."""
    mg = generate_edge_ordered_tournament_infinite_limit(4)
    mm = minimax(mg)[0]
    return int(mg.copeland_scores()[mm] == -3)

def process_real_election_profile(prof, use_extended_strict_preference=True):
    """Process a single profile and classify its 4-candidate subprofiles."""
    try:
        if use_extended_strict_preference:
            prof.use_extended_strict_preference()
        
        mg = prof.margin_graph()
        results = []
        
        # Pre-compute all margins for efficiency and error prevention
        margins = {}
        for a in prof.candidates:
            for b in prof.candidates:
                if a != b:
                    try:
                        margins[(a,b)] = mg.margin(a,b)
                    except Exception as e:
                        print(f"Warning: Error computing margin for {a},{b}: {str(e)}")
                        margins[(a,b)] = 0
        
        for subset in itertools.combinations(prof.candidates, 4):
            # Skip if there are any zero margins
            has_zero = False
            for a in subset:
                for b in subset:
                    if a != b and margins[(a,b)] == 0:
                        has_zero = True
                        break
                if has_zero:
                    break
            
            if has_zero:
                continue
            
            # Extract the underlying directed graph
            g = nx.DiGraph()
            g.add_nodes_from(subset)
            directed_edges = [(a,b) for a,b in margins.keys() if a in subset and b in subset and margins[(a,b)] > 0]
            g.add_edges_from(directed_edges)
            
            result = {
                'linear_order': 0,
                'bottom_cycle': 0,
                'ascending_top_cycle': 0,
                'descending_top_cycle': 0,
                'SL_four_cycle': 0,
                'LS_four_cycle': 0,
                'num_voters': prof.num_voters
            }
            
            # Classify the tournament
            if nx.is_isomorphic(g, linear_order):
                result['linear_order'] = 1
            elif nx.is_isomorphic(g, bottom_cycle):
                result['bottom_cycle'] = 1
            elif nx.is_isomorphic(g, top_cycle):
                top_cycle_cands = copeland(mg)
                if len(top_cycle_cands) != 3:
                    continue
                    
                # Find candidate with largest win
                max_margin_in_top_cycle = max([margins[(c1,c2)] for c1 in top_cycle_cands for c2 in top_cycle_cands if c1 != c2])
                top_cycle_cands_with_largest_win = [c for c in top_cycle_cands if max([margins[(c,c2)] for c2 in top_cycle_cands if c != c2]) == max_margin_in_top_cycle]
                
                if len(top_cycle_cands_with_largest_win) != 1:
                    continue
                    
                top_cycle_cand_with_largest_win = top_cycle_cands_with_largest_win[0]
                
                # Find candidate with smallest loss
                min_pos_margin_in_top_cycle = min([margins[(c1,c2)] for c1 in top_cycle_cands for c2 in top_cycle_cands if c1 != c2 and margins[(c1,c2)] > 0])
                top_cycle_cands_with_smallest_loss = [c for c in top_cycle_cands if min([margins[(c2,c)] for c2 in top_cycle_cands if c2 != c and margins[(c2,c)] > 0]) == min_pos_margin_in_top_cycle]
                
                if len(top_cycle_cands_with_smallest_loss) != 1:
                    continue
                    
                top_cycle_cand_with_smallest_loss = top_cycle_cands_with_smallest_loss[0]
                
                if top_cycle_cand_with_smallest_loss != top_cycle_cand_with_largest_win:
                    result['ascending_top_cycle'] = 1
                else:
                    result['descending_top_cycle'] = 1
            elif nx.is_isomorphic(g, four_cycle):
                copeland_winners = copeland(mg)
                if len(copeland_winners) != 2:
                    continue
                    
                margin_between_copeland_winners = max([margins[(c1,c2)] for c1 in copeland_winners for c2 in copeland_winners if c1 != c2]) 
                margin_of_non_copeland_over_copeland = max([margins[(c,c2)] for c in mg.candidates if c not in copeland_winners for c2 in copeland_winners])
                
                if margin_of_non_copeland_over_copeland < margin_between_copeland_winners:
                    result['SL_four_cycle'] = 1
                else:
                    result['LS_four_cycle'] = 1
            
            results.append(result)
    except Exception as e:
        print(f"Error processing profile: {str(e)}")
        return []
    
    return results

def analyze_dataset(profiles):
    """Analyze a dataset of profiles and print statistics."""
    all_results = []
    for prof in tqdm(profiles):
        if len(prof.candidates) >= 4:
            results = process_real_election_profile(prof)
            all_results.extend(results)
    
    if not all_results:
        print("No valid 4-candidate subprofiles found.")
        return
    
    # Calculate statistics
    total_subprofiles = len(all_results)
    totals = {
        'linear_order': 0,
        'bottom_cycle': 0,
        'ascending_top_cycle': 0,
        'descending_top_cycle': 0,
        'SL_four_cycle': 0,
        'LS_four_cycle': 0
    }
    
    num_voters_list = [r['num_voters'] for r in all_results]
    
    for result in all_results:
        for key in totals:
            totals[key] += result[key]
    
    print(f"\nDataset Statistics:")
    print(f"Total number of 4-candidate subprofiles with no zero margins: {total_subprofiles}")
    print(f"Average number of voters: {np.mean(num_voters_list):.1f}")
    print("\nTournament type frequencies:")
    for tournament_type, count in totals.items():
        percentage = (count / total_subprofiles) * 100
        print(f"{tournament_type.replace('_', ' ').title()}: {percentage:.2f}% ({count} out of {total_subprofiles})")

def main():
    """Main function to run all analyses."""
    # Initialize multiprocessing pool
    num_cpus = max(1, os.cpu_count() - 1)  # Leave one core free
    pool = Pool(num_cpus)
    
    try:
        # Section 1: Tournament statistics
        num_trials = 10_000  # Reduced for testing
        print(f"\nSection 1: Processing {num_trials:,} trials using {num_cpus} CPU cores...")
        results = list(tqdm(pool.imap(process_single_tournament, range(num_trials)), total=num_trials))
        
        # Aggregate results
        totals = {
            'linear_order': 0,
            'bottom_cycle': 0,
            'ascending_top_cycle': 0,
            'descending_top_cycle': 0,
            'SL_four_cycle': 0,
            'LS_four_cycle': 0
        }
        
        for result in results:
            for key in totals:
                totals[key] += result[key]
        
        # Print results
        print("\nResults:")
        for tournament_type, count in totals.items():
            percentage = (count / num_trials) * 100
            print(f"{tournament_type.replace('_', ' ').title()}: {percentage:.2f}%")
            
        # Section 2: Method comparisons
        num_trials = 1_000  # Reduced for testing
        print(f"\nSection 2: Comparing methods across {num_trials:,} trials...")
        results = list(tqdm(pool.imap(compare_methods, range(num_trials)), total=num_trials))
        
        # Aggregate results
        totals = {method: 0 for method in ['rp', 'mm', 'smm', 'bp', 'sv', 'rv']}
        for result in results:
            for method in totals:
                totals[method] += result[method]
        
        # Print results
        print("\nFrequency of disagreement with:")
        method_names = {
            'bp': 'Beat Path',
            'mm': 'Minimax',
            'rp': 'Ranked Pairs',
            'rv': 'River',
            'smm': 'Smith Minimax',
            'sv': 'Stable Voting'
        }
        
        for method, count in sorted(totals.items(), key=lambda x: method_names[x[0]]):
            percentage = (count / num_trials) * 100
            print(f"{method_names[method]}: {percentage:.2f}%")
            
        # Section 3: Heavy top cycles
        num_trials = 1_000  # Reduced for testing
        print(f"\nSection 3: Checking heavy top cycles across {num_trials:,} trials...")
        results = list(tqdm(pool.imap(check_heavy_top_cycle, range(num_trials)), total=num_trials))
        
        # Calculate total and percentage
        heavy_top_cycle_count = sum(results)
        percentage = (heavy_top_cycle_count / num_trials) * 100
        print(f"\nHeavy top cycle: {percentage:.2f}%")
        
        # Section 4: Dataset processing
        print("\nSection 4: Processing datasets...")
        
        # Process PrefLib dataset (test subset)
        print("\nProcessing PrefLib dataset (test subset)...")
        preflib_profiles = []
        elections = []
        
        # Process only first 5 files of each type for testing
        for ext in ['.soi', '.toi', '.toc']:
            files = glob.glob(f"real_elections/preflib_dataset/*{ext}")[:5]
            print(f"Found {len(files)} {ext} files for testing")
            for fname in files:
                try:
                    election_name = fname.split("/")[-1].split(".")[0]
                    if election_name not in elections:
                        print(f"Processing {fname}...")
                        elections.append(election_name)
                        profile = ProfileWithTies.read(fname)
                        if len(profile.candidates) >= 4:
                            preflib_profiles.append(profile)
                            print(f"Added profile with {len(profile.candidates)} candidates")
                        else:
                            print(f"Skipped profile with only {len(profile.candidates)} candidates")
                except Exception as e:
                    print(f"Error processing {fname}: {str(e)}")
                    continue
        
        print(f"\nProcessing {len(preflib_profiles)} valid PrefLib profiles...")
        analyze_dataset(preflib_profiles)
        
        # Process Otis 2022 dataset (test subset)
        print("\nProcessing Otis 2022 dataset (test subset)...")
        items_to_skip = ['skipped', 'overvote', 'undervote']
        otis_profiles = []
        
        # Process only first 5 zip files for testing
        files = glob.glob("real_elections/otis_2022_dataset/*.zip")[:5]
        print(f"Found {len(files)} zip files for testing")
        
        for file in files:
            try:
                print(f"Processing {file}...")
                with ZipFile(file, 'r') as zip_ref:
                    csv_files = [name for name in zip_ref.namelist() if name.endswith(".csv")][:5]
                    print(f"Processing {len(csv_files)} CSV files from {file}")
                    for name in csv_files:
                        try:
                            with zip_ref.open(name) as f:
                                csv_bytes = f.read()
                                csv_text = csv_bytes.decode('utf-8')
                                csv_buffer = io.StringIO(csv_text)
                                prof = ProfileWithTies.read(
                                    csv_buffer,
                                    file_format='csv',
                                    csv_format='rank_columns',
                                    items_to_skip=items_to_skip
                                )
                                if len(prof.candidates) >= 4:
                                    otis_profiles.append(prof)
                                    print(f"Added profile with {len(prof.candidates)} candidates from {name}")
                                else:
                                    print(f"Skipped profile with only {len(prof.candidates)} candidates from {name}")
                        except Exception as e:
                            print(f"Error processing CSV {name} in {file}: {str(e)}")
                            continue
            except Exception as e:
                print(f"Error processing zip file {file}: {str(e)}")
                continue
        
        print(f"\nProcessing {len(otis_profiles)} valid Otis profiles...")
        analyze_dataset(otis_profiles)
        
    finally:
        # Clean up
        pool.close()
        pool.join()

if __name__ == '__main__':
    main()

# Section 2: Table 1 - frequency of disagreement with other methods (Parallelized)

# End of file
