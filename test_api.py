import requests
import json

url = "https://seasonal-api.open-meteo.com/v1/seasonal?latitude=38.9072&longitude=-77.0369&daily=temperature_2m_mean,temperature_2m_max,temperature_2m_min"
resp = requests.get(url)
print(json.dumps(resp.json(), indent=2))
