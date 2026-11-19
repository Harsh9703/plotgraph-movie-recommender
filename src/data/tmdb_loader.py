"""TMDB and The Movies Dataset metadata ingestion, cleaning, linking, and coverage reporting.

Phase 3A: Links MovieLens-1M movies safely to TMDB metadata using explicit IDs from links.csv.
Preserves all MovieLens catalog movies without dropping unmapped items.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def safe_parse_literal(val: Any) -> Any:
    """Safely parse python literal structures using ast.literal_eval.

    Never evaluates arbitrary code or executes unsafe operations.

    Args:
        val: Input value, which may be a string, dict, list, NaN, or None.

    Returns:
        Parsed Python object, or None if invalid/empty/malformed.
    """
    if val is None or pd.isna(val):
        return None
    if isinstance(val, (list, dict)):
        return val
    if not isinstance(val, str):
        return None

    val_str = val.strip()
    if not val_str or val_str.lower() in ("nan", "none", ""):
        return None

    try:
        return ast.literal_eval(val_str)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None


def extract_release_decade(date_val: Any) -> Optional[str]:
    """Extract release decade string (e.g., '1990s') from a date string or timestamp.

    Args:
        date_val: Date string (e.g. '1995-10-30', '1995') or date object.

    Returns:
        Decade string formatted like '1990s', or None if invalid/missing.
    """
    if date_val is None or pd.isna(date_val):
        return None

    date_str = str(date_val).strip()
    if not date_str:
        return None

    match = re.match(r"^(\d{4})", date_str)
    if not match:
        return None

    try:
        year = int(match.group(1))
        # Validate reasonable cinema release year range
        if 1880 <= year <= 2030:
            decade = (year // 10) * 10
            return f"{decade}s"
    except (ValueError, OverflowError):
        pass

    return None


def parse_genres_field(genres_val: Any) -> Optional[List[str]]:
    """Parse TMDB genres string into a list of genre names.

    Args:
        genres_val: String representation of genre dicts (e.g. [{'id': 16, 'name': 'Animation'}]).

    Returns:
        List of genre name strings, or None if empty/invalid.
    """
    parsed = safe_parse_literal(genres_val)
    if isinstance(parsed, list):
        genres = [
            str(g.get("name")).strip()
            for g in parsed
            if isinstance(g, dict) and g.get("name") and str(g.get("name")).strip()
        ]
        if genres:
            return genres
    return None


def parse_collection_field(coll_val: Any) -> Optional[str]:
    """Parse TMDB belongs_to_collection string into collection name.

    Args:
        coll_val: String representation of collection dict.

    Returns:
        Collection name string, or None if missing/invalid.
    """
    parsed = safe_parse_literal(coll_val)
    if isinstance(parsed, dict):
        name = parsed.get("name")
        if name and str(name).strip():
            return str(name).strip()
    return None


def extract_director(crew_val: Any) -> Optional[str]:
    """Extract the primary director name from crew list.

    Args:
        crew_val: String representation of crew members.

    Returns:
        Director's name string, or None if missing.
    """
    parsed = safe_parse_literal(crew_val)
    if isinstance(parsed, list):
        for member in parsed:
            if isinstance(member, dict):
                job = str(member.get("job", "")).strip().lower()
                if job == "director":
                    name = member.get("name")
                    if name and str(name).strip():
                        return str(name).strip()
    return None


def extract_top_actors(cast_val: Any, top_k: int = 5) -> Optional[List[str]]:
    """Extract up to top-K billed actor names ordered by billing order.

    Args:
        cast_val: String representation of cast members.
        top_k: Maximum number of top-billed actors to retain.

    Returns:
        List of actor name strings, or None if empty.
    """
    parsed = safe_parse_literal(cast_val)
    if isinstance(parsed, list):
        # Sort by billing order (lower order = higher billing)
        sorted_cast = sorted(
            parsed,
            key=lambda x: (
                x.get("order", 999)
                if isinstance(x, dict) and isinstance(x.get("order"), (int, float))
                else 999
            ),
        )
        actors = []
        for member in sorted_cast:
            if isinstance(member, dict):
                name = member.get("name")
                if name and str(name).strip():
                    actors.append(str(name).strip())
                if len(actors) >= top_k:
                    break
        if actors:
            return actors
    return None


def extract_keywords(kw_val: Any) -> Optional[List[str]]:
    """Extract keyword names from TMDB keywords list.

    Args:
        kw_val: String representation of keyword dicts.

    Returns:
        List of keyword strings, or None if empty.
    """
    parsed = safe_parse_literal(kw_val)
    if isinstance(parsed, list):
        keywords = [
            str(item.get("name")).strip()
            for item in parsed
            if isinstance(item, dict) and item.get("name") and str(item.get("name")).strip()
        ]
        if keywords:
            return keywords
    return None


def clean_tmdb_metadata(
    metadata_input: Union[str, Path, pd.DataFrame],
    tmdb_ids_filter: Optional[Set[int]] = None,
) -> pd.DataFrame:
    """Load and clean movies_metadata.csv from The Movies Dataset.

    Handles real-world quirks defensively:
    - Drops malformed non-numeric ID rows (e.g. Kaggle rows with shifted dates in ID column).
    - Deduplicates by tmdb_id deterministically, keeping higher vote_count and complete overview.
    - Parses genres, belongs_to_collection, release_decade safely.
    - Cleans overview, popularity, vote_average, vote_count.

    Args:
        metadata_input: File path or raw DataFrame.
        tmdb_ids_filter: Optional set of TMDB IDs to filter immediately for speed.

    Returns:
        Cleaned DataFrame with deduplicated TMDB metadata.
    """
    if isinstance(metadata_input, (str, Path)):
        df = pd.read_csv(metadata_input, low_memory=False)
    else:
        df = metadata_input.copy()

    # Drop non-numeric IDs
    valid_id_mask = df["id"].astype(str).str.isnumeric()
    df = df[valid_id_mask].copy()
    df["tmdb_id"] = df["id"].astype(int)

    # Optional fast filter
    if tmdb_ids_filter is not None:
        df = df[df["tmdb_id"].isin(tmdb_ids_filter)].copy()

    # Numeric cleaning for sorting and deduplication
    df["vote_count_clean"] = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0).astype(float)
    df["overview_len"] = df["overview"].fillna("").astype(str).str.strip().str.len()
    df["orig_idx"] = df.index

    # Deterministic deduplication: prefer highest vote_count, longest overview, lowest index
    df = df.sort_values(
        by=["tmdb_id", "vote_count_clean", "overview_len", "orig_idx"],
        ascending=[True, False, False, True],
    ).drop_duplicates(subset=["tmdb_id"], keep="first")

    # Clean fields
    df["tmdb_title"] = df["title"].apply(
        lambda x: str(x).strip() if pd.notna(x) and str(x).strip() else None
    )
    df["overview"] = df["overview"].apply(
        lambda x: str(x).strip()
        if pd.notna(x) and str(x).strip() and str(x).strip().lower() != "none"
        else None
    )
    df["release_date"] = df["release_date"].apply(
        lambda x: str(x).strip() if pd.notna(x) and str(x).strip() else None
    )
    df["release_decade"] = df["release_date"].apply(extract_release_decade)
    df["tmdb_genres"] = df["genres"].apply(parse_genres_field)
    df["collection_name"] = df["belongs_to_collection"].apply(parse_collection_field)
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").astype(float)
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").astype(float)
    df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce").astype("Int64")

    cols = [
        "tmdb_id",
        "tmdb_title",
        "overview",
        "release_date",
        "release_decade",
        "tmdb_genres",
        "collection_name",
        "popularity",
        "vote_average",
        "vote_count",
    ]
    return df[cols].reset_index(drop=True)


def parse_credits(
    credits_input: Union[str, Path, pd.DataFrame],
    top_k_actors: int = 5,
    tmdb_ids_filter: Optional[Set[int]] = None,
) -> pd.DataFrame:
    """Load and parse credits.csv, extracting director and top-K actors.

    Args:
        credits_input: File path or raw DataFrame.
        top_k_actors: Number of top-billed cast members to extract.
        tmdb_ids_filter: Optional set of TMDB IDs to filter immediately.

    Returns:
        DataFrame with columns: ['tmdb_id', 'director', 'top_actors'].
    """
    if isinstance(credits_input, (str, Path)):
        df = pd.read_csv(credits_input, low_memory=False)
    else:
        df = credits_input.copy()

    valid_id_mask = df["id"].astype(str).str.isnumeric()
    df = df[valid_id_mask].copy()
    df["tmdb_id"] = df["id"].astype(int)

    if tmdb_ids_filter is not None:
        df = df[df["tmdb_id"].isin(tmdb_ids_filter)].copy()

    # Deduplicate deterministically
    df = df.drop_duplicates(subset=["tmdb_id"], keep="first")

    df["director"] = df["crew"].apply(extract_director)
    df["top_actors"] = df["cast"].apply(lambda x: extract_top_actors(x, top_k=top_k_actors))

    cols = ["tmdb_id", "director", "top_actors"]
    return df[cols].reset_index(drop=True)


def parse_keywords(
    keywords_input: Union[str, Path, pd.DataFrame],
    tmdb_ids_filter: Optional[Set[int]] = None,
) -> pd.DataFrame:
    """Load and parse keywords.csv, extracting clean keyword string lists.

    Args:
        keywords_input: File path or raw DataFrame.
        tmdb_ids_filter: Optional set of TMDB IDs to filter immediately.

    Returns:
        DataFrame with columns: ['tmdb_id', 'keywords'].
    """
    if isinstance(keywords_input, (str, Path)):
        df = pd.read_csv(keywords_input, low_memory=False)
    else:
        df = keywords_input.copy()

    valid_id_mask = df["id"].astype(str).str.isnumeric()
    df = df[valid_id_mask].copy()
    df["tmdb_id"] = df["id"].astype(int)

    if tmdb_ids_filter is not None:
        df = df[df["tmdb_id"].isin(tmdb_ids_filter)].copy()

    # Deduplicate deterministically
    df = df.drop_duplicates(subset=["tmdb_id"], keep="first")

    df["keywords"] = df["keywords"].apply(extract_keywords)

    cols = ["tmdb_id", "keywords"]
    return df[cols].reset_index(drop=True)


def link_movielens_tmdb(
    movies_df: pd.DataFrame,
    links_df: pd.DataFrame,
    clean_meta_df: pd.DataFrame,
    clean_credits_df: pd.DataFrame,
    clean_keywords_df: pd.DataFrame,
) -> pd.DataFrame:
    """Link MovieLens movies to TMDB metadata using links.csv.

    Crucial safety rules:
    - Join strictly on explicit IDs (movieId -> tmdbId -> tmdb_id). No fuzzy joins.
    - Retains ALL MovieLens catalog movies (3,883). Never drops unlinked movies.
    - Preserves exact internal movie_idx and movie_id mappings.
    - Marks availability via boolean flags (has_tmdb_link, has_overview, has_credits, has_keywords).

    Args:
        movies_df: Processed MovieLens movies (movie_id, movie_idx, title, genres).
        links_df: links.csv from The Movies Dataset (movieId, tmdbId).
        clean_meta_df: Cleaned metadata DataFrame from clean_tmdb_metadata.
        clean_credits_df: Cleaned credits DataFrame from parse_credits.
        clean_keywords_df: Cleaned keywords DataFrame from parse_keywords.

    Returns:
        Unified DataFrame with all MovieLens catalog movies and TMDB metadata fields.
    """
    # Prepare links: drop invalid movieId, convert to int
    links_clean = links_df.dropna(subset=["movieId"]).copy()
    links_clean["movieId"] = links_clean["movieId"].astype(int)
    links_clean = links_clean.drop_duplicates(subset=["movieId"], keep="first")

    # Step 1: Merge MovieLens catalog with links
    merged = movies_df.copy()
    merged = merged.merge(
        links_clean[["movieId", "tmdbId"]],
        left_on="movie_id",
        right_on="movieId",
        how="left",
    )
    # Convert tmdbId to nullable Int64
    merged["tmdb_id"] = pd.to_numeric(merged["tmdbId"], errors="coerce").astype("Int64")

    # Step 2: Merge with TMDB metadata on tmdb_id
    merged = merged.merge(
        clean_meta_df,
        on="tmdb_id",
        how="left",
    )

    # Step 3: Merge with credits on tmdb_id
    merged = merged.merge(
        clean_credits_df,
        on="tmdb_id",
        how="left",
    )

    # Step 4: Merge with keywords on tmdb_id
    merged = merged.merge(
        clean_keywords_df,
        on="tmdb_id",
        how="left",
    )

    # Format output fields
    merged["movie_id"] = merged["movie_id"].astype(int)
    merged["movie_idx"] = merged["movie_idx"].astype(int)
    merged["ml_title"] = merged["title"].astype(str)
    merged["ml_genres"] = merged["genres"].astype(str)

    # Boolean flags
    merged["has_tmdb_link"] = merged["tmdb_id"].notna()
    merged["has_overview"] = (
        merged["overview"].notna()
        & (merged["overview"].astype(str).str.strip() != "")
        & (merged["overview"].astype(str).str.strip().str.lower() != "none")
    )
    merged["has_credits"] = merged["director"].notna() | merged["top_actors"].apply(
        lambda x: isinstance(x, list) and len(x) > 0
    )
    merged["has_keywords"] = merged["keywords"].apply(
        lambda x: isinstance(x, list) and len(x) > 0
    )

    # Enforce exact requested columns
    ordered_cols = [
        "movie_id",
        "movie_idx",
        "ml_title",
        "ml_genres",
        "tmdb_id",
        "tmdb_title",
        "overview",
        "release_date",
        "release_decade",
        "tmdb_genres",
        "collection_name",
        "director",
        "top_actors",
        "keywords",
        "popularity",
        "vote_average",
        "vote_count",
        "has_tmdb_link",
        "has_overview",
        "has_credits",
        "has_keywords",
    ]

    result = merged[ordered_cols].sort_values(by="movie_idx").reset_index(drop=True)

    # Safety assertions: ensure no catalog movies were dropped or duplicated
    assert len(result) == len(movies_df), (
        f"Catalog size mismatch: expected {len(movies_df)}, got {len(result)}"
    )
    assert list(result["movie_id"]) == list(movies_df["movie_id"]), (
        "movie_id sequence does not match original movies_df!"
    )
    assert list(result["movie_idx"]) == list(movies_df["movie_idx"]), (
        "movie_idx sequence does not match original movies_df!"
    )

    return result


def compute_coverage_report(
    unified_df: pd.DataFrame,
    rated_movie_ids: Optional[Set[int]] = None,
    candidate_movie_idxs: Optional[Set[int]] = None,
) -> Dict[str, Any]:
    """Compute detailed coverage metrics across catalog, rated, and candidate movies.

    Args:
        unified_df: Unified movie metadata DataFrame.
        rated_movie_ids: Set of movie_ids appearing in user interactions.
        candidate_movie_idxs: Set of movie_idxs appearing in top-100 candidate pools.

    Returns:
        Dictionary with coverage statistics and percentages for each group.
    """
    def _calc_stats(subset: pd.DataFrame, name: str) -> Dict[str, Any]:
        total = len(subset)
        if total == 0:
            return {"name": name, "total_movies": 0}

        has_link = int(subset["has_tmdb_link"].sum())
        has_meta = int((subset["has_tmdb_link"] & subset["tmdb_title"].notna()).sum())
        has_overview = int(subset["has_overview"].sum())
        has_credits = int(subset["has_credits"].sum())
        has_director = int(subset["director"].notna().sum())
        has_actors = int(
            subset["top_actors"].apply(lambda x: isinstance(x, list) and len(x) > 0).sum()
        )
        has_keywords = int(subset["has_keywords"].sum())
        has_collection = int(subset["collection_name"].notna().sum())

        return {
            "name": name,
            "total_movies": total,
            "with_tmdb_link": has_link,
            "with_tmdb_link_pct": round(has_link / total * 100.0, 2),
            "with_metadata": has_meta,
            "with_metadata_pct": round(has_meta / total * 100.0, 2),
            "with_overview": has_overview,
            "with_overview_pct": round(has_overview / total * 100.0, 2),
            "with_credits": has_credits,
            "with_credits_pct": round(has_credits / total * 100.0, 2),
            "with_director": has_director,
            "with_director_pct": round(has_director / total * 100.0, 2),
            "with_actors": has_actors,
            "with_actors_pct": round(has_actors / total * 100.0, 2),
            "with_keywords": has_keywords,
            "with_keywords_pct": round(has_keywords / total * 100.0, 2),
            "with_collection": has_collection,
            "with_collection_pct": round(has_collection / total * 100.0, 2),
        }

    report: Dict[str, Any] = {
        "catalog": _calc_stats(unified_df, "All Catalog Movies"),
    }

    if rated_movie_ids is not None:
        rated_subset = unified_df[unified_df["movie_id"].isin(rated_movie_ids)]
        report["rated_movies"] = _calc_stats(rated_subset, "Rated Interaction Movies")

    if candidate_movie_idxs is not None:
        cand_subset = unified_df[unified_df["movie_idx"].isin(candidate_movie_idxs)]
        report["candidate_movies"] = _calc_stats(cand_subset, "Top-100 Candidate Movies")

    return report


def build_unified_metadata_pipeline(
    config_or_path: Union[str, Path, Dict[str, Any]],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute the end-to-end metadata cleaning and linking pipeline.

    Args:
        config_or_path: Path to metadata.yaml or loaded dictionary.

    Returns:
        Tuple of (unified_df, coverage_report).
    """
    if isinstance(config_or_path, (str, Path)):
        with open(config_or_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = config_or_path

    paths = config["paths"]
    params = config.get("parameters", {})
    top_k_actors = params.get("top_k_actors", 5)

    # 1. Load MovieLens movies
    movies_parquet_path = Path(paths["ml1m_movies_parquet"])
    if not movies_parquet_path.exists():
        raise FileNotFoundError(f"MovieLens movies file not found at {movies_parquet_path}")
    movies_df = pd.read_parquet(movies_parquet_path)
    logger.info(f"Loaded {len(movies_df)} MovieLens-1M movies.")

    # 2. Load links.csv
    links_path = Path(paths["links_csv"])
    if not links_path.exists():
        raise FileNotFoundError(f"links.csv not found at {links_path}")
    links_df = pd.read_csv(links_path)
    logger.info(f"Loaded {len(links_df)} links from {links_path}.")

    # Identify TMDB IDs linked to our MovieLens catalog for fast selective loading
    links_clean = links_df.dropna(subset=["movieId", "tmdbId"]).copy()
    links_clean["movieId"] = links_clean["movieId"].astype(int)
    ml_links = movies_df.merge(links_clean, left_on="movie_id", right_on="movieId", how="inner")
    mapped_tmdb_ids = set(ml_links["tmdbId"].astype(int).unique())
    logger.info(f"Found {len(mapped_tmdb_ids)} unique TMDB IDs mapped to MovieLens catalog.")

    # 3. Clean movies_metadata.csv
    meta_path = Path(paths["movies_metadata_csv"])
    if not meta_path.exists():
        raise FileNotFoundError(f"movies_metadata.csv not found at {meta_path}")
    logger.info(f"Cleaning metadata from {meta_path}...")
    clean_meta = clean_tmdb_metadata(meta_path, tmdb_ids_filter=mapped_tmdb_ids)

    # 4. Clean credits.csv
    credits_path = Path(paths["credits_csv"])
    if not credits_path.exists():
        raise FileNotFoundError(f"credits.csv not found at {credits_path}")
    logger.info(f"Parsing credits from {credits_path}...")
    clean_credits = parse_credits(credits_path, top_k_actors=top_k_actors, tmdb_ids_filter=mapped_tmdb_ids)

    # 5. Clean keywords.csv
    keywords_path = Path(paths["keywords_csv"])
    if not keywords_path.exists():
        raise FileNotFoundError(f"keywords.csv not found at {keywords_path}")
    logger.info(f"Parsing keywords from {keywords_path}...")
    clean_keywords = parse_keywords(keywords_path, tmdb_ids_filter=mapped_tmdb_ids)

    # 6. Link everything
    logger.info("Linking MovieLens movies with TMDB metadata...")
    unified_df = link_movielens_tmdb(
        movies_df=movies_df,
        links_df=links_df,
        clean_meta_df=clean_meta,
        clean_credits_df=clean_credits,
        clean_keywords_df=clean_keywords,
    )

    # 7. Collect rated movie IDs and candidate movie IDs for coverage reporting
    rated_movie_ids: Optional[Set[int]] = None
    candidate_movie_idxs: Optional[Set[int]] = None

    splits_dir = Path(paths.get("splits_dir", ""))
    if splits_dir.exists():
        split_dfs = []
        for split_name in ("train.parquet", "val.parquet", "test.parquet"):
            sp_path = splits_dir / split_name
            if sp_path.exists():
                split_dfs.append(pd.read_parquet(sp_path))
        if split_dfs:
            all_splits = pd.concat(split_dfs, ignore_index=True)
            rated_movie_ids = set(all_splits["movie_id"].unique())

    top100_path = Path(paths.get("top100_candidates_parquet", ""))
    if top100_path.exists():
        top100_df = pd.read_parquet(top100_path)
        candidate_movie_idxs = set(top100_df["movie_idx"].unique())

    coverage_report = compute_coverage_report(
        unified_df=unified_df,
        rated_movie_ids=rated_movie_ids,
        candidate_movie_idxs=candidate_movie_idxs,
    )

    # 8. Save outputs
    output_parquet_path = Path(paths["output_parquet"])
    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    unified_df.to_parquet(output_parquet_path, engine="pyarrow", index=False)
    logger.info(f"Saved unified metadata to {output_parquet_path}")

    report_json_path = Path(paths["coverage_report_json"])
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(coverage_report, f, indent=2)
    logger.info(f"Saved coverage report to {report_json_path}")

    return unified_df, coverage_report
