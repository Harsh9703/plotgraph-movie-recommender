# PlotGraph Sample Metadata & Summaries

This directory contains lightweight, committed summary artifacts illustrating dataset coverage and metadata schemas without committing large binary or raw tables:

1. **`coverage_report.json`**:
   - Comprehensive audit of TMDB metadata coverage across the 3,883 MovieLens-1M catalog movies.
   - Summarizes linkage rates, overview availability (3,810 movies / 98.12%), director coverage (97.94%), cast coverage (97.04%), keywords coverage (86.07%), and franchise collection memberships (17.23%).
2. **`sample_summary.json`**:
   - Deterministic 30-movie sample summary used during prompt development and pilot extraction.
   - Illustrates the merged schema between MovieLens catalog identifiers and TMDB metadata fields.
