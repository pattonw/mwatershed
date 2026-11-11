window.BENCHMARK_DATA = {
  "lastUpdate": 1794440375397,
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
      }
    ]
  }
}