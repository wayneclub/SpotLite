Here is the updated `CLI.md` tailored to the new Python code provided.

Major changes include:

1. **Configuration**: Added instructions for `api_keys.json`.
2. **Workflow Split**: The single `analyze` command has been replaced by a two-phase workflow: `kewords` (Analysis) and `summary` (LLM Generation).
3. **New Arguments**: Added documentation for `--domain` in the keyword analysis phase.

---

# SpotLite CLI Guide

The SpotLite CLI provides a unified interface to:

* Scrape **Google Maps reviews** using Selenium
* Fetch **place details** via the Google Places API
* Run **Phase 1: Keyword & Sentiment Analysis** (NLP)
* Run **Phase 2: AI Summary Generation** (LLM)

All commands are run from the project root.

---

# 1. Installation

## Python Version

SpotLite supports:

* **Python 3.11+** (recommended)

## Install Dependencies

```bash
pip install -r requirements.txt

```

## Setup Configuration

SpotLite now uses split configuration files.

* **API Keys** (`api_keys.json`):

Create `api_keys.json` in the project root for credentials:
```json
{
  "google_maps": {
    "api_key": "YOUR_GOOGLE_MAPS_API_KEY",
    "timeout_seconds": 20
  }
}

```



---

# 2. CLI Usage Overview

Run CLI:

```bash
python cli.py <command> [options]

```

Available commands:

| Command | Phase | Description |
| --- | --- | --- |
| `reviews` | Scraper | Scrape Google Maps reviews |
| `details` | Scraper | Fetch Google Maps place details |
| `kewords` | Analysis (Phase 1) | Extract keywords, calculate TF-IDF, output CSV |
| `summary` | Analysis (Phase 2) | Generate LLM summaries from analyzed JSON |
| `-d` / `--debug` | Global | Enable debug logs + file logging |

---

# 3. Command: `reviews`

Scrape reviews for a single Google Maps place URL using Selenium.

## Usage

```bash
python cli.py reviews -u "<google_maps_url>" [-d]

```

## Options

| Option | Description |
| --- | --- |
| `-u`, `--url` | **Required**. Google Maps place link. |
| `-d`, `--debug` | Enable debug logging + save logs to file. |

## Output

Saves to `data/reviews/`:

* `<place_name>_reviews.json`
* `<place_name>_reviews.csv`

---

# 4. Command: `details`

Fetch structured place details via Google Places API. Requires `api_keys.json` to be configured.

## Usage

```bash
python cli.py details -u "<google_maps_url>" [-d]

```

## Behavior

* Extracts `place_id` from the URL.
* Queries Google Places API using the key in `api_keys.json`.
* Saves structured JSON.

## Output

Typical output:

```
data/details/<place_id>.json

```

---

# 5. Command: `kewords` (Phase 1)

**Note:** The command is strictly `kewords` (as defined in the code), aimed at "Keywords Extraction".

This command performs **Phase 1** of the analysis pipeline:

1. Loads raw review JSON.
2. Extracts phrases and sentiment using `AspectKeywordAnalyzer`.
3. Computes TF-IDF statistics.
4. Outputs CSV reports.

## Usage

```bash
python cli.py kewords -i <path_to_reviews.json> [--domain <domain>] [-d]

```

## Options

| Option | Description |
| --- | --- |
| `-i`, `--input` | **Required**. Path to the raw `reviews.json` file. |
| `--domain` | Domain context for NLP (default: `restaurant`). |
| `-d`, `--debug` | Enable debug logging. |

## Output

Generates analysis files in the same directory or `outputs/` (depending on configuration), typically including:

* Keyword statistics (CSV)
* Sentiment analysis results

---

# 6. Command: `summary` (Phase 2)

This command performs **Phase 2** of the analysis pipeline:

1. Loads the analyzed data (JSON format).
2. Passes data to the LLM module (`generate_review_summary`).
3. Generates a natural language summary.

## Usage

```bash
python cli.py summary -i <path_to_analyzed_data.json> [-d]

```

## Options

| Option | Description |
| --- | --- |
| `-i`, `--input` | **Required**. Path to the JSON file containing top5_keyword.json data. |
| `-d`, `--debug` | Enable debug logging. |

---

# 7. Logging System

Logging is controlled by `configs.json` and the `-d` flag.

* **Default**: Level INFO, logs to console.
* **Debug Mode (`-d`)**: Level DEBUG, logs to file (defaults to `logs/app.log`).

To log inside modules:

```python
import logging
logger = logging.getLogger(__name__)

logger.info("Scraping started")

```

---

# 8. Examples

## Full Workflow Example

**1. Scrape Reviews**

```bash
python cli.py reviews -u "https://goo.gl/maps/example"

```

**2. Analyze Keywords (Phase 1)**

```bash
python cli.py kewords -i data/reviews/MyPlace_reviews.json --domain restaurant

```

**3. Generate AI Summary (Phase 2)**

```bash
# Assuming Phase 1 generated a specific JSON output
python cli.py summary -i outputs/MyPlace_top5_keywords.json

```

---

# End of CLI Guide