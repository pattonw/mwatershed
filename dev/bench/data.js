window.BENCHMARK_DATA = {
  "lastUpdate": 1794596456606,
  "repoUrl": "https://github.com/pattonw/mwatershed",
  "entries": {
    "mwatershed": [
      {
        "commit": {
          "author": {
            "name": "William Patton",
            "username": "pattonw",
            "email": "wllmpttn24@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "1797201835448ff82fbc811fdd03b6488d82a913",
          "message": "Merge pull request #13 from pattonw/docs-deploy-on-tags\n\nDeploy docs on version tags and by hand only",
          "timestamp": "2026-11-11T23:38:15Z",
          "url": "https://github.com/pattonw/mwatershed/commit/1797201835448ff82fbc811fdd03b6488d82a913"
        },
        "date": 1794440374788,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "import mwatershed (cold)",
            "value": 81.142,
            "range": "± 8.114",
            "unit": "ms",
            "extra": "interpreter startup 11.2 ms subtracted; n=10"
          },
          {
            "name": "pytest wall time",
            "value": 10.166,
            "unit": "s",
            "extra": "`uv run pytest` in the tests job"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "William Patton",
            "username": "pattonw",
            "email": "wllmpttn24@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "36262d0d27defcf9ff60a30b6efd47958cc3232c",
          "message": "Merge pull request #14 from pattonw/docs/warnings-fatal\n\nMake Sphinx warnings fatal and fix the current warnings",
          "timestamp": "2026-11-13T18:59:48Z",
          "url": "https://github.com/pattonw/mwatershed/commit/36262d0d27defcf9ff60a30b6efd47958cc3232c"
        },
        "date": 1794596455951,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "import mwatershed (cold)",
            "value": 55.537,
            "range": "± 0.518",
            "unit": "ms",
            "extra": "interpreter startup 7.9 ms subtracted; n=10"
          },
          {
            "name": "pytest wall time",
            "value": 7.281,
            "unit": "s",
            "extra": "`uv run pytest` in the tests job"
          }
        ]
      }
    ]
  }
}