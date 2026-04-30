# AI Service Search Improvement Plan

## Status: Not Started

### 1. [ ] Create TODO.md (Done)
### 2. [x] Improve Intent Detection in ExpertServiceSearch.py
   - Expand intent_examples with service queries
   - Add 'service_query' intent  
   - Lower INTENT_THRESHOLD to 0.22
### 3. [x] Install Dependencies (BM25 already available, rapidfuzz used)
### 4. [ ] Test Changes
   - Run `python test_search.py`
   - Verify higher intent confidence (>0.3 for service queries)
   - Test interactive mode
### 5. [ ] Complete & Demo

