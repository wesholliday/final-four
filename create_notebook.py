import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Add cells
cells = []

# Title and description
cells.append(nbf.v4.new_markdown_cell("""# Final Four Tournament Analysis (Parallelized Version)
This notebook is a parallelized version of final-four.ipynb, supplementary material for the paper 
"A simple Condorcet voting method for Final Four elections" by Wesley H. Holliday."""))

# Imports
cells.append(nbf.v4.new_code_cell("""import os
import pref_voting
from pref_voting.profiles import *
from pref_voting.voting_methods import *
from pref_voting.combined_methods import *
from pref_voting.margin_based_methods import *
from pref_voting.generate_profiles import generate_profile 
from pref_voting.generate_weighted_majority_graphs import generate_edge_ordered_tournament_infinite_limit
from pref_voting.profiles_with_ties import ProfileWithTies
from pref_voting.io.readers import preflib_to_profile

from pathos.multiprocessing import ProcessingPool as Pool
from tqdm.notebook import tqdm
import networkx as nx
import numpy as np
import glob
from zipfile import ZipFile
import io
import itertools
from itertools import combinations
from preflibtools.instances import OrdinalInstance

# Configure parallel processing
num_cpus = max(1, os.cpu_count() - 1)
pool = Pool(num_cpus)

# Print versions for reference
print(f"pref_voting version: {pref_voting.__version__}")
print(f"Using {num_cpus} CPU cores for parallel processing")"""))

# Section 1 header and setup
cells.append(nbf.v4.new_markdown_cell("""# 1. Figure 1 percentages
First we create networkx digraphs for the four tournament isomorphism types."""))

cells.append(nbf.v4.new_code_cell("""# Create the tournament types
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
four_cycle.add_edges_from([(0,1),(1,2), (2,3), (3,0), (2,0), (1,3)])"""))

# Proposed method definition
cells.append(nbf.v4.new_code_cell("""@vm(name = "Proposed Method")
def proposed_method(edata, curr_cands = None):
    copeland_winners = copeland(edata)
    worst_loss = dict()
    
    for c in copeland_winners:
        worst_loss[c] = max([edata.margin(c2, c) for c2 in edata.candidates])
    
    smallest_worst_loss = min(worst_loss.values())
    copeland_winners_with_smallest_worst_loss = [c for c in copeland_winners if worst_loss[c] == smallest_worst_loss]
    
    return copeland_winners_with_smallest_worst_loss"""))

# Section 1 main loop
cells.append(nbf.v4.new_markdown_cell("""## Tournament Statistics (Parallelized)
We'll analyze tournament statistics using parallel processing to improve performance."""))

cells.append(nbf.v4.new_code_cell("""def process_tournament(_):
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

# Run parallel trials
num_trials = 100_000  # Reduced for testing
print(f"Running {num_trials:,} trials in parallel...")

results = pool.map(process_tournament, range(num_trials))

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

# Print percentages
for tournament_type, count in totals.items():
    percentage = (count / num_trials) * 100
    print(f"{tournament_type}: {percentage:.2f}%")"""))

# Section 2 header
cells.append(nbf.v4.new_markdown_cell("""# 2. Table 1 - Frequency of disagreement with other methods"""))

cells.append(nbf.v4.new_code_cell("""def compare_methods(_):
    mg = generate_edge_ordered_tournament_infinite_limit(4)
    
    pm = proposed_method(mg)
    rp = ranked_pairs(mg)
    mm = minimax(mg)
    smm = smith_minimax(mg)
    bp = beat_path(mg)
    sv = stable_voting(mg)
    rv = river(mg)
    
    return {
        'rp': pm != rp,
        'mm': pm != mm,
        'smm': pm != smm,
        'bp': pm != bp,
        'sv': pm != sv,
        'rv': pm != rv
    }

# Run parallel trials
num_trials = 10_000  # Reduced for testing
print(f"Running {num_trials:,} trials in parallel...")

results = pool.map(compare_methods, range(num_trials))

# Aggregate results
totals = {
    'rp': 0, 'mm': 0, 'smm': 0,
    'bp': 0, 'sv': 0, 'rv': 0
}

for result in results:
    for method in totals:
        if result[method]:
            totals[method] += 1

# Print percentages
method_names = {
    'bp': 'Beat Path',
    'mm': 'Minimax',
    'rp': 'Ranked Pairs',
    'rv': 'River',
    'smm': 'Smith Minimax',
    'sv': 'Stable Voting'
}

for method, count in totals.items():
    percentage = (count / num_trials) * 100
    print(f"Frequency of disagreement with {method_names[method]}: {percentage:.2f}%")"""))

# Section 3 header
cells.append(nbf.v4.new_markdown_cell("""# 3. Footnote 27 - Frequency of heavy top cycles"""))

cells.append(nbf.v4.new_code_cell("""def check_heavy_top_cycle(_):
    mg = generate_edge_ordered_tournament_infinite_limit(4)
    mm = minimax(mg)[0]
    return 1 if mg.copeland_scores()[mm] == -3 else 0

# Run parallel trials
num_trials = 10_000  # Reduced for testing
print(f"Running {num_trials:,} trials in parallel...")

results = pool.map(check_heavy_top_cycle, range(num_trials))

# Calculate percentage
heavy_top_cycle_count = sum(results)
percentage = (heavy_top_cycle_count / num_trials) * 100
print(f"Heavy top cycle: {percentage:.2f}%")"""))

# Add all cells to notebook
nb['cells'] = cells

# Write the notebook to a file
with open('final-four-parallel.ipynb', 'w') as f:
    nbf.write(nb, f)
