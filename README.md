# Toronto Maple Leafs — 2025/2026 Season Predictor 🏒

This is a tool to predict the outcome of the upcoming 2025/26 season for **your Toronto Maple Leafs!**

## The Basics 📕

This tool uses an elo-style model to predict outcomes across multiple simulations. Each NHL team has been assigned an Elo-style rating (more on that later). This elo-rating represents their skill-level relative to the rest of the league.

For the Toronto Maple leafs in particular, their 2025 offseason transactions (most notably the departure of Mitch Marner) are factored into their elo rating).

## The Elo Rating 📈

What goes into each elo rating (at a glance):

### Overall Team Rating

- Even-strength (5v5) — net expected goals per 60: xGF/60 − xGA/60
- Power play (5v4) — xGF/60 (higher is better)
- Penalty kill (4v5) — xGA/60 (lower is better; we invert it)
- Goaltending — team save percentage (SV%)
  - For this one in particular, I went for team percentage as a team's starter may not always be playing

> P.S. If you're thinking "what the heck are expected goals?". You can read all about this stat [here](https://insidetherink.com/inside-the-stats-expected-goals/)

### Game-by-Game Basis

- Home ice: +HOME_EDGE Elo to the home team (default +5)
- Back-to-back: −B2B_PENALTY Elo if playing on consecutive days (default −10)
- Rest: REST_PTS_PER_DAY × rest_diff Elo (default +3 per day)

## Adjustable Elements 🔧

There are a number of tunable elements of the dashboard that can affect the results of the simulation. Feel free to customize each!

 - Number of simulations
 - Random seed vs. fixed seed
 - Addition of elo noise (to factor in chance/randomness)

Elo Specific Adjustments:
 - Home Bonus (Teams often perform better at home)
 - Back-to-back penalty (Teams often perform worse on a back-to-back)
 - Rest day bonus (Teams often perform better after rest)
 - Elo Weighting (again, to factor in chance/randomness)

## Downloadables ⬇️

There are two .csv files available to download.
 - Per-game Predictions: Contains the table on the site, showing the predicition for each game of the season
 - Simulations: Shows the record for each simulation run (depends on number of simulations value in sidebar)

## Insights 🧠

The fun part: what did I learn?

For starters, hockey is a very, very, random sport. It's part of what makes it so amazing. Unfortuantely for a project like this, that leaves a lot up to non-quantifiable fields. As I ran backtests on last seasons data, I found myself saying "well thats just ludicrous, they would never lose that game" only to check against the data and see they actually did (the leafs have a habit of that...). Furthermore, the leafs even last season were not analyitically out-of-this-world. Yet, they finished first in their division. So there's just some level of intangibles that are difficult to capture in a project like this.

Currently, here are features that I would like to add:
- Further backtesting on older seasons schedules (the problem being that it requires a new elo for each team, each season due to roster changes)
- Increased emphasis on player potential (i.e. younger teams will probably surpass previous expectations)
- More statistics factored into elo (hits, blocks, possession time etc.)
