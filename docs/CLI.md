# SpotLite CLI Guide

The SpotLite CLI provides a unified interface to:

- Scrape **Google Maps reviews** using Selenium
- Fetch **place details** via the Google Places API
- Run **NLP keyword analysis** on scraped review data

All commands are run from the project root.

---

# 1. Installation

## Python Version
SpotLite supports:

- **Python 3.11+** (recommended)

## Install Dependencies

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Setup Configuration

```bash
cp configs.example.json configs.json
```

Update your Google API key:

```jsonc
"google_maps": {
  "api_key": "YOUR_API_KEY",
  "timeout_seconds": 20
}
```

---

# 2. CLI Usage Overview

Run CLI:

```bash
python cli.py <command> [options]
```

Available commands:

| Command   | Description |
|-----------|-------------|
| `reviews` | Scrape Google Maps reviews |
| `details` | Fetch Google Maps place details |
| `analyze` | Run NLP keyword analysis on reviews JSON |
| `-d` / `--debug` | Enable debug logs + file logging |

---

# 3. Command: `reviews`

Scrape reviews for a single Google Maps place URL using Selenium.

## Usage

```bash
python cli.py reviews -u "<google_maps_url>" [-d]
```

## Options

| Option | Description |
|--------|-------------|
| `-u`, `--url` | **Required**. Google Maps place link. |
| `-d`, `--debug` | Enable debug logging + save logs to file. |

## Features

- Automatically expands short links (`maps.app.goo.gl`, `goo.gl/maps`)
- Forces English UI in Selenium
- Automatically clicks:
  - **Sort → Newest**
  - Every **See more** button
- Smart scrolling logic:
  - Stops at configured limits
  - Stops when encountering reviews older than `stop_at_years_ago` (e.g., `2 years ago`)

## Output

Configured in:

```jsonc
"reviews": {
  "output_root": "data/reviews",
  "stop_at_years_ago": 2,
  "save_json": true,
  "save_csv": true
}
```

Saves:

- `<place_name>_reviews.json`
- `<place_name>_reviews.csv`

---

# 4. Command: `details`

Fetch structured place details via Google Places API.

## Usage

```bash
python cli.py details -u "<google_maps_url>" [-d]
```

## Behavior

- Resolves short links if needed
- Extracts `place_id` or `text_query`
- Requests details via Google Maps Places API
- Saves pretty-formatted JSON

## Output Config

```jsonc
"details": {
  "output_root": "data/details",
  "save_json": true
}
```

Typical output:

```
data/details/<place_id>.json
```

---

# 5. Command: `analyze`

Run NLP keyword extraction / clustering analysis.

## Usage

```bash
python cli.py analyze -i path/to/reviews.json [-d]
```

## Options

| Option | Description |
|--------|-------------|
| `-i`, `--input` | Path to a reviews JSON file |
| `-d`, `--debug` | Enable debug logging |

## Behavior

- Loads JSON
- Runs analysis (`spotlite.analysis.keywords`)
- Saves output under `outputs/` or as configured

---

# 6. Logging System

Logging is controlled by `configs.json`:

```jsonc
"logging": {
  "level": "INFO",
  "console": true,
  "save_to_file": false,
  "file_path": "logs/app.log"
}
```

When you pass `-d` / `--debug`:

- Logging level switches to **DEBUG**
- Logs are always written to file (even if config says false)

To log inside modules:

```python
import logging
logger = logging.getLogger(__name__)

logger.info("Scraping started")
logger.debug("Raw HTML length = %d", len(html))
```

---

# 7. Examples

## Scrape reviews
```bash
python cli.py reviews -u "https://maps.app.goo.gl/EXAMPLE"
```

## Scrape reviews with debug logs
```bash
python cli.py reviews -u "https://maps.app.goo.gl/EXAMPLE" -d
```

## Fetch place details
```bash
python cli.py details -u "https://maps.app.goo.gl/EXAMPLE"
```

## Analyze review JSON
```bash
python cli.py analyze -i data/reviews/SomePlace_reviews.json
```

---

# 8. Future CLI Extensions

Planned:

- Batch scraping:
  ```bash
  python cli.py reviews-batch -i places.csv
  ```
- Write results to database:
  ```bash
  python cli.py sync-db -i data/reviews/*.json
  ```
- AI summary generation:
  ```bash
  python cli.py summarize -i <reviews.json>
  ```

The CLI is intentionally thin—heavy logic remains inside the `spotlite` Python package for maximum maintainability.

---

# End of CLI Guide
