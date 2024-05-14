# Topic Modeling for Grouped Data

## Overview
This project uses advanced natural language processing techniques to analyze and compare textual data across different segments of a dataset. It consists of three main scripts that execute Latent Dirichlet Allocation (LDA) topic modeling, calculate cosine similarities between topic distributions of different subsets, and apply author modeling to explore how topics are distributed across the full dataset. This project is particularly useful for understanding thematic structures in large text corpora and discovering both unique and shared topics across different data segments.

## Project Structure
- **LDA Topic Modeling**: Performs topic modeling on the entire dataset to identify and characterize major themes.
- **Cosine Similarity Analysis**: Computes the similarity between topic distributions from two distinct subsets of the dataset, providing insights into their thematic overlap.
- **Segment-Based Author Modeling**: Analyzes how different data segments contribute to the overall topic distributions, providing a unique perspective on the influence and specificity of each segment.

### Files
1. `LDA.py` - Script for performing LDA topic modeling on the entire dataset.
2. `LDA-multi.py` - Script for calculating cosine similarities between the topic distributions of two data subsets.
3. `author-tm.py` - Script for implementing author modeling to assess topic distributions across the dataset.
