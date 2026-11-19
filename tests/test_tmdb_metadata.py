"""Unit and integration tests for Phase 3A: TMDB metadata ingestion and MovieLens linking."""

from pathlib import Path
import pandas as pd
import pytest

from src.data.tmdb_loader import (
    safe_parse_literal,
    extract_release_decade,
    parse_genres_field,
    parse_collection_field,
    extract_director,
    extract_top_actors,
    extract_keywords,
    clean_tmdb_metadata,
    parse_credits,
    parse_keywords,
    link_movielens_tmdb,
    compute_coverage_report,
    build_unified_metadata_pipeline,
)


def test_safe_parse_literal_safety_and_types():
    """Verify safe_parse_literal parses valid structures and rejects malformed/hostile input."""
    # Valid dict and list strings
    assert safe_parse_literal("[{'id': 1, 'name': 'Animation'}]") == [{"id": 1, "name": "Animation"}]
    assert safe_parse_literal("{'id': 10, 'name': 'Toy Story'}") == {"id": 10, "name": "Toy Story"}

    # Numbers and booleans
    assert safe_parse_literal("123") == 123
    assert safe_parse_literal("True") is True

    # None, NaN, and whitespace
    assert safe_parse_literal(None) is None
    assert safe_parse_literal(float("nan")) is None
    assert safe_parse_literal("   ") is None
    assert safe_parse_literal("nan") is None
    assert safe_parse_literal("None") is None

    # Already parsed Python objects
    assert safe_parse_literal([{"name": "Tom Hanks"}]) == [{"name": "Tom Hanks"}]
    assert safe_parse_literal({"key": "val"}) == {"key": "val"}

    # Malformed strings (should not raise)
    assert safe_parse_literal("[{'id': 1, 'name':") is None
    assert safe_parse_literal("{incomplete") is None

    # Hostile code execution attempt (ast.literal_eval must reject and return None)
    assert safe_parse_literal("__import__('os').system('echo pwned')") is None


def test_extract_release_decade():
    """Verify release decade derivation across various formats and boundary cases."""
    assert extract_release_decade("1995-10-30") == "1990s"
    assert extract_release_decade("2001-05-18") == "2000s"
    assert extract_release_decade("2019-12-25") == "2010s"
    assert extract_release_decade("1980-01-01") == "1980s"
    assert extract_release_decade("1977") == "1970s"
    assert extract_release_decade("1927-01-10") == "1920s"

    # Out of cinema bounds
    assert extract_release_decade("1750-01-01") is None
    assert extract_release_decade("2050-01-01") is None

    # Missing / malformed
    assert extract_release_decade("") is None
    assert extract_release_decade(None) is None
    assert extract_release_decade(float("nan")) is None
    assert extract_release_decade("not-a-date") is None


def test_genres_and_collection_parsing():
    """Verify genres and collection extraction from nested structures."""
    genres_raw = "[{'id': 16, 'name': 'Animation'}, {'id': 35, 'name': 'Comedy'}]"
    assert parse_genres_field(genres_raw) == ["Animation", "Comedy"]

    # Empty genres
    assert parse_genres_field("[]") is None
    assert parse_genres_field(None) is None
    assert parse_genres_field("malformed") is None

    # Collection parsing
    coll_raw = "{'id': 10194, 'name': 'Toy Story Collection'}"
    assert parse_collection_field(coll_raw) == "Toy Story Collection"
    assert parse_collection_field("nan") is None
    assert parse_collection_field(None) is None


def test_director_and_actor_extraction():
    """Verify director identification and top-K actor extraction in billing order."""
    crew_raw = (
        "[{'job': 'Producer', 'name': 'Bonnie Arnold'}, "
        "{'job': 'Director', 'name': 'John Lasseter'}, "
        "{'job': 'Writer', 'name': 'Pete Docter'}]"
    )
    assert extract_director(crew_raw) == "John Lasseter"

    # Case-insensitive director matching
    crew_lower = "[{'job': 'director', 'name': 'Jane Campion'}]"
    assert extract_director(crew_lower) == "Jane Campion"

    # No director
    assert extract_director("[{'job': 'Producer', 'name': 'Someone'}]") is None
    assert extract_director(None) is None

    # Top actors ordered by 'order'
    cast_raw = (
        "[{'name': 'Don Rickles', 'order': 2}, "
        "{'name': 'Tom Hanks', 'order': 0}, "
        "{'name': 'Tim Allen', 'order': 1}, "
        "{'name': 'Jim Varney', 'order': 3}]"
    )
    top2 = extract_top_actors(cast_raw, top_k=2)
    assert top2 == ["Tom Hanks", "Tim Allen"]

    top3 = extract_top_actors(cast_raw, top_k=3)
    assert top3 == ["Tom Hanks", "Tim Allen", "Don Rickles"]

    # Empty cast
    assert extract_top_actors("[]", top_k=5) is None
    assert extract_top_actors(None, top_k=5) is None


def test_keywords_extraction():
    """Verify keyword string list extraction."""
    kw_raw = "[{'id': 931, 'name': 'jealousy'}, {'id': 4290, 'name': 'toy'}]"
    assert extract_keywords(kw_raw) == ["jealousy", "toy"]
    assert extract_keywords("[]") is None
    assert extract_keywords(None) is None


def test_clean_tmdb_metadata_rejection_and_deduplication():
    """Verify that clean_tmdb_metadata rejects non-numeric IDs and deduplicates by vote_count."""
    df_raw = pd.DataFrame({
        "id": [
            "862",         # Valid movie
            "1997-08-20",  # Malformed row (shifted date in ID column)
            "100",         # Duplicate movie, lower vote count
            "100",         # Duplicate movie, higher vote count
        ],
        "title": [
            "Toy Story",
            "Malformed Entry",
            "Duplicate Low Votes",
            "Duplicate High Votes",
        ],
        "overview": [
            "Led by Woody...",
            "Bad row",
            "Short overview",
            "Complete detailed overview of the movie",
        ],
        "release_date": ["1995-10-30", "1", "2000-01-01", "2000-01-01"],
        "genres": [
            "[{'id': 16, 'name': 'Animation'}]",
            "[]",
            "[]",
            "[{'id': 18, 'name': 'Drama'}]",
        ],
        "belongs_to_collection": ["{'name': 'Toy Story Collection'}", None, None, None],
        "popularity": ["21.94", "bad", "5.0", "15.0"],
        "vote_average": ["7.7", "bad", "6.0", "7.5"],
        "vote_count": ["5415", "bad", "50", "500"],
    })

    cleaned = clean_tmdb_metadata(df_raw)

    # Malformed non-numeric ID must be dropped
    assert "1997-08-20" not in cleaned["tmdb_id"].values
    assert len(cleaned) == 2  # IDs: 862 and 100

    # ID 100 duplicate check: higher vote count (500) must be retained
    row_100 = cleaned[cleaned["tmdb_id"] == 100].iloc[0]
    assert row_100["tmdb_title"] == "Duplicate High Votes"
    assert row_100["vote_count"] == 500
    assert row_100["release_decade"] == "2000s"
    assert row_100["tmdb_genres"] == ["Drama"]

    # ID 862 check
    row_862 = cleaned[cleaned["tmdb_id"] == 862].iloc[0]
    assert row_862["tmdb_title"] == "Toy Story"
    assert row_862["collection_name"] == "Toy Story Collection"
    assert row_862["release_decade"] == "1990s"


def test_link_movielens_tmdb_preserves_unmapped_movies():
    """Verify that link_movielens_tmdb retains all MovieLens movies even without TMDB links."""
    movies_df = pd.DataFrame({
        "movie_id": [1, 2, 3],
        "movie_idx": [0, 1, 2],
        "title": ["Toy Story (1995)", "Jumanji (1995)", "Shadows (1988)"],
        "genres": ["Animation|Children's", "Adventure|Children's", "Drama"],
    })

    links_df = pd.DataFrame({
        "movieId": [1, 2, 3],
        "tmdbId": [862.0, 8844.0, float("nan")],  # Movie 3 has no TMDB ID
    })

    meta_df = pd.DataFrame({
        "tmdb_id": [862, 8844],
        "tmdb_title": ["Toy Story", "Jumanji"],
        "overview": ["Toys come alive", ""],  # Movie 2 has empty overview
        "release_date": ["1995-10-30", "1995-12-15"],
        "release_decade": ["1990s", "1990s"],
        "tmdb_genres": [["Animation"], ["Adventure"]],
        "collection_name": ["Toy Story Collection", None],
        "popularity": [21.9, 17.0],
        "vote_average": [7.7, 6.9],
        "vote_count": [5415, 2413],
    })

    credits_df = pd.DataFrame({
        "tmdb_id": [862],  # Movie 2 has no credits record
        "director": ["John Lasseter"],
        "top_actors": [["Tom Hanks", "Tim Allen"]],
    })

    keywords_df = pd.DataFrame({
        "tmdb_id": [862, 8844],
        "keywords": [["toy", "friendship"], ["board game"]],
    })

    unified = link_movielens_tmdb(
        movies_df=movies_df,
        links_df=links_df,
        clean_meta_df=meta_df,
        clean_credits_df=credits_df,
        clean_keywords_df=keywords_df,
    )

    # All 3 catalog movies MUST be preserved
    assert len(unified) == 3
    assert list(unified["movie_id"]) == [1, 2, 3]
    assert list(unified["movie_idx"]) == [0, 1, 2]

    # Movie 1: Fully linked
    m1 = unified.iloc[0]
    assert m1["has_tmdb_link"] == True
    assert m1["has_overview"] == True
    assert m1["has_credits"] == True
    assert m1["has_keywords"] == True
    assert m1["director"] == "John Lasseter"
    assert m1["collection_name"] == "Toy Story Collection"

    # Movie 2: Linked but empty overview and missing credits
    m2 = unified.iloc[1]
    assert m2["has_tmdb_link"] == True
    assert m2["has_overview"] == False  # Empty string overview
    assert m2["has_credits"] == False   # No credits in table
    assert m2["has_keywords"] == True

    # Movie 3: Completely unmapped movie (must remain in catalog!)
    m3 = unified.iloc[2]
    assert m3["movie_id"] == 3
    assert m3["movie_idx"] == 2
    assert m3["ml_title"] == "Shadows (1988)"
    assert pd.isna(m3["tmdb_id"])
    assert m3["has_tmdb_link"] == False
    assert m3["has_overview"] == False
    assert m3["has_credits"] == False
    assert m3["has_keywords"] == False


def test_coverage_reporting_metrics():
    """Verify that compute_coverage_report correctly calculates counts and percentages."""
    unified_df = pd.DataFrame({
        "movie_id": [1, 2, 3, 4],
        "movie_idx": [0, 1, 2, 3],
        "has_tmdb_link": [True, True, True, False],
        "tmdb_title": ["Movie 1", "Movie 2", "Movie 3", None],
        "has_overview": [True, True, False, False],
        "has_credits": [True, False, False, False],
        "director": ["Director 1", None, None, None],
        "top_actors": [["Actor A"], None, None, None],
        "has_keywords": [True, True, False, False],
        "collection_name": ["Col 1", None, None, None],
    })

    report = compute_coverage_report(
        unified_df=unified_df,
        rated_movie_ids={1, 2},
        candidate_movie_idxs={0},
    )

    cat = report["catalog"]
    assert cat["total_movies"] == 4
    assert cat["with_tmdb_link"] == 3
    assert cat["with_tmdb_link_pct"] == 75.0
    assert cat["with_overview"] == 2
    assert cat["with_overview_pct"] == 50.0

    rated = report["rated_movies"]
    assert rated["total_movies"] == 2
    assert rated["with_tmdb_link"] == 2
    assert rated["with_tmdb_link_pct"] == 100.0

    cand = report["candidate_movies"]
    assert cand["total_movies"] == 1
    assert cand["with_credits"] == 1
    assert cand["with_credits_pct"] == 100.0


def test_real_data_metadata_pipeline():
    """Integration test verifying end-to-end output against real local files."""
    parquet_path = Path("data/processed/ml-1m/metadata/movies_with_tmdb.parquet")
    if not parquet_path.exists():
        pytest.skip("Phase 3A output parquet not generated yet")

    df = pd.read_parquet(parquet_path)

    # 1. Catalog integrity
    assert len(df) == 3883, f"Expected 3883 MovieLens movies, got {len(df)}"
    assert df["movie_id"].is_unique
    assert df["movie_idx"].is_unique
    assert list(df["movie_idx"]) == list(range(3883))

    # 2. MovieLens title and genre completeness
    assert df["ml_title"].isna().sum() == 0
    assert df["ml_genres"].isna().sum() == 0

    # 3. Known exact coverage counts
    assert df["has_tmdb_link"].sum() == 3828
    assert (~df["has_tmdb_link"]).sum() == 55
    assert df["has_overview"].sum() == 3810

    # 4. Check Toy Story (movie_id=1)
    toy_story = df[df["movie_id"] == 1].iloc[0]
    assert toy_story["movie_idx"] == 0
    assert toy_story["tmdb_id"] == 862
    assert toy_story["tmdb_title"] == "Toy Story"
    assert toy_story["director"] == "John Lasseter"
    assert "Tom Hanks" in toy_story["top_actors"]
    assert toy_story["release_decade"] == "1990s"
    assert toy_story["collection_name"] == "Toy Story Collection"
    assert toy_story["has_overview"] == True

    # 5. Check unmapped movie (movie_id=142)
    unmapped = df[df["movie_id"] == 142].iloc[0]
    assert pd.isna(unmapped["tmdb_id"])
    assert unmapped["has_tmdb_link"] == False
    assert unmapped["has_overview"] == False
    assert unmapped["has_credits"] == False
