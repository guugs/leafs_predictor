import requests
import pandas as pd

def get_schedule(team='tor', season='20252026'):
    url = f'https://api-web.nhle.com/v1/club-schedule-season/{team}/{season}'
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()
    games = data.get('games', [])
    df = pd.DataFrame(games)
    return df

if __name__ == "__main__":
    df = get_schedule()
    print(df.head())
